import random
import os
import h5py
import numpy as np

from dlabalone.ablboard import GameState
from dlabalone.abltypes import Player
from dlabalone.data.populate_games import GamePopulator
from dlabalone.rl.experience import ExperienceCollector
from dlabalone.utils import encode_board_str


def _get_filename_with_num(orig_name):
    basename = os.path.basename(orig_name)
    if basename.count('.') > 0:
        dot_index = orig_name.rfind('.')
        return orig_name[:dot_index]+'_%d'+orig_name[dot_index:]
    else:
        return orig_name + '_%d'


def _fill_collectors(collector_list, orig_name, populator):
    if populator is not None:
        name_format = _get_filename_with_num(orig_name)
        for i in range(6):
            collector_list.append(ExperienceCollector(name_format % i))
    else:
        collector_list.append(ExperienceCollector(orig_name))


class ExperienceGameState(object):
    def __init__(self, encoder_black, encoder_white, h5file_black=None, h5file_white=None, populator=None):
        self.collectors_black = []
        self.collectors_white = []
        if h5file_black is not None:
            _fill_collectors(self.collectors_black, h5file_black, populator)
            for collector in self.collectors_black:
                collector.begin_episode()

        if h5file_white is not None:
            _fill_collectors(self.collectors_white, h5file_white, populator)
            for collector in self.collectors_white:
                collector.begin_episode()
        self.collectors = {Player.black: self.collectors_black,
                           Player.white: self.collectors_white}
        self.encoder = {Player.black: encoder_black,
                        Player.white: encoder_white}
        self.populator = populator
        self.stat = {'total_moves': 0, 'new_moves': 0, 'winner': None,
                     'h5files': (list(map(lambda x: x.h5file_name, self.collectors_black)),
                                 list(map(lambda x: x.h5file_name, self.collectors_white)))}
        self.game_state = GameState.new_game(5)
        self.step_count = 0
        self.past_boards = {}
        self.too_many_repeat = False

    def add_stat(self, action, move_probs):
        self.stat['total_moves'] += 1
        if action != np.argmax(move_probs):
            self.stat['new_moves'] += 1

    def apply_move(self, move, encoded, estimated_value, move_probs):
        self.step_count += 1
        next_player = self.game_state.next_player
        action = self.encoder[next_player].encode_move(move)
        for collector in self.collectors[next_player]:
            collector.record_decision(encoded, action, estimated_value)
            encoded = self.populator.rotate_numpy_encoded(encoded)
            move = self.populator.rotate_move(move, 1)
            action = self.encoder[next_player].encode_move(move)
        self.add_stat(action, move_probs)
        self.game_state.apply_move_lite(move)

        # Check draw
        board_str = encode_board_str(self.game_state.board)
        count = self.past_boards.get(board_str, 0) + 1
        self.past_boards[board_str] = count
        if count >= 30:
            self.too_many_repeat = True

    def save_result(self):
        if self.game_state.winner() == Player.black:
            reward_black = 1
            reward_white = -1
        elif self.game_state.winner() == Player.white:
            reward_black = -1
            reward_white = 1
        else:
            return

        for collector in self.collectors[Player.black]:
            collector.complete_episode(reward_black)
            collector.save_as_file()
        for collector in self.collectors[Player.white]:
            collector.complete_episode(reward_white)
            collector.save_as_file()

    def get_stat(self):
        self.stat['winner'] = self.game_state.winner()
        return self.stat

    def is_draw(self):
        if self.step_count >= 2000:
            return True
        return self.too_many_repeat

    # Functions from GameStat
    @property
    def board(self):
        return self.game_state.board

    @property
    def next_player(self):
        return self.game_state.next_player

    def legal_moves(self):
        return self.game_state.legal_moves()

    def is_over(self):
        return self.game_state.is_over()


def experience_simulation(num_games,
                          models_black, models_white,
                          encoder_black, encoder_white,
                          move_selector_black, move_selector_white,
                          predict_convertor_black, predict_convertor_white,
                          exp_dir=None, black_name='black', white_name='white', populate_games=True):
    # Save parameters
    h5file_black = None
    h5file_white = None
    if exp_dir:
        randkey = random.random()
        h5file_black = f'experience_{randkey}_%03d_{black_name}_black.h5'
        h5file_white = f'experience_{randkey}_%03d_{white_name}_white.h5'
        h5file_black = os.path.join(exp_dir, h5file_black)
        h5file_white = os.path.join(exp_dir, h5file_white)
    encoder = {Player.black: encoder_black, Player.white: encoder_white}
    models = {Player.black: models_black, Player.white: models_white}
    predict_convertor = {Player.black: predict_convertor_black, Player.white: predict_convertor_white}
    move_selector = {Player.black: move_selector_black, Player.white: move_selector_white}
    if populate_games:
        populator = GamePopulator(5)
    else:
        populator = None

    # Initialize variables
    stat_list = []
    next_player = Player.black

    # Initialize games
    game_list = []
    for index in range(num_games):
        if exp_dir:
            fd_black = h5py.File(h5file_black % index, 'w')
            fd_white = h5py.File(h5file_white % index, 'w')
        else:
            fd_black = None
            fd_white = None
        game_list.append(ExperienceGameState(encoder_black, encoder_white,
                                             h5file_black=fd_black, h5file_white=fd_white, populator=populator))

    #####################
    # Run games
    #####################
    while True:
        # Get encoded boards
        predict_input = np.zeros((len(game_list),) + encoder[next_player].shape())
        for index, game in enumerate(game_list):
            encoded = encoder[next_player].encode_board(game.board, game.next_player)
            predict_input[index] = encoded

        # Predict
        predict_output_batch_list = []
        for model in models[next_player]:
            predict_output_batch_list.append(model.predict_on_batch(predict_input))

        # Select move & apply
        for index, game in enumerate(game_list):
            move_probs, estimated_value = predict_convertor[next_player](predict_output_batch_list, index)
            move = move_selector[next_player](encoder[next_player], move_probs, game.legal_moves())
            game.apply_move(move, predict_input[index], estimated_value, move_probs)

        # Remove finished games
        finished_games = []
        for game in game_list:
            if game.is_over():
                finished_games.append(game)
                game.save_result()
                stat_list.append(game.get_stat())
            elif game.is_draw():
                finished_games.append(game)
        for game in finished_games:
            game_list.remove(game)

        if len(game_list) == 0:
            break
        next_player = next_player.other
    return stat_list
