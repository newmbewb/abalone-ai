import random
import os
import h5py


class SimulateGame():
    def __init__(self, h5file_black=None, h5file_white=None):
        if h5file_black:
            self.collector_black = ExperienceCollector()
        if h5file_white:
            self.collector_white = ExperienceCollector()
        self.state = GameState.new_game(5)
    def apply_move(self, move):


def _experience_simulation(num_games, models_black, models_white, encoder_black, encoder_white, exp_dir=None, black_name='black', white_name='white', game_per_file=1, populate_games=True):
    if exp_dir:
        randkey = random.random()
        h5file_black = f'experience_{randkey}_%03d_{black_name}_black.h5'
        h5file_white = f'experience_{randkey}_%03d_{white_name}_white.h5'
        h5file_black = os.path.join(exp_dir, h5file_black)
        h5file_white = os.path.join(exp_dir, h5file_white)
#        fd_black = h5py.File(os.path.join(exp_dir, h5file_black), 'w')
#        fd_white = h5py.File(os.path.join(exp_dir, h5file_white), 'w')
    # Initialize variables
    win_count = {Player.black: 0, Player.white: 0}

    # Initialize games
    game_list = []
    for white in range(num_games):
        if exp_dir:
            fd_black = h5py.File(h5file_black % index, 'w')
            fd_white = h5py.File(h5file_white % index, 'w')
        else:
            fd_black = None
            fd_white = None
        game_list.append(SimulateGame(h5file_black=fd_black, f5file_white=fd_white))

    #####################
    # Run games
    #####################

    # Get encoded boards
    predict_input = np.zeros((len(game_list),) + cur_encoder.shape())
    for index, game in enumerate(game_list):
        encoded = cur_encoder.encode_board(game.board, game.next_player)
        predict_input[index] = encoded

    # Predict
    predict_output = cur_model.predict_on_batch(predict_input)

    # Select move & apply
    for index, game in enumerate(game_list):
        single_output = predict_output[index]
        move = cur_move_selector(encoder, single_output, game.legal_moves())
        game.apply_move_lite(move, predict_input[index], single_output)

    # Remove finished games
    finished_games = []
    for game in game_list:
        if game.is_over():
            finished_games.append(game)
            game.save_result()
            win_count[game.winner()] += 1
    for game in finished_games:
        game_list.remove(game)

#_experience_simulation(1, 1, 1, 1, 1, '33')
