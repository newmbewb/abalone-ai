import gc
import os
import shutil
from pathlib import Path

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.optimizers import SGD

from dlabalone.data.data_processor import DataGenerator
from dlabalone.data.generate_dataset import generate_dataset
from dlabalone.data.populate_games import GamePopulator
from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.networks import ac_simple1_v2
from dlabalone.networks.base import prepare_tf_custom_objects
from dlabalone.rl_mcts.game_generator import generate_games
from dlabalone.rl_mcts.winrate_evaluator import evaluate_winrates

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RLMCTSTrainLogger:
    def __init__(self, home):
        self.log_file = os.path.join(home, 'rl_mcts_train_log.txt')
        # Variable for state
        self.encode_done = False
        self.exclude_file_list = []
        if os.path.exists(self.log_file):
            self._load_log()
        else:
            self.generation = 1
            self.next_stage = 1

    def _load_log(self):
        # Init state values
        self.encode_done = False
        self.exclude_file_list = []

        # Read log file
        fd = open(self.log_file)
        generation, next_stage = fd.readline().strip().split(',')
        self.generation = int(generation)
        self.next_stage = int(next_stage)

        if self.next_stage == 2 or self.next_stage == 4:
            is_encoded_data_ready = fd.readline().strip()
            if is_encoded_data_ready == 'True':
                self.encode_done = True
            elif is_encoded_data_ready == 'False':
                self.encode_done = False
            else:
                assert False, 'Something wrong'
        elif self.next_stage == 1 or self.next_stage == 3:
            self.exclude_file_list = fd.readline().strip().split(',')
        else:
            assert False, 'Something wrong'

    def _save_log(self):
        fd = open(self.log_file, 'w')
        fd.write(f'{self.generation},{self.next_stage}\n')
        data = []
        if self.next_stage == 1:
            fd.write(','.join(self.exclude_file_list) + '\n')
        elif self.next_stage == 2:
            data.append(self.encode_done)
            fd.write(','.join(map(str, data)) + '\n')
        elif self.next_stage == 3:
            fd.write(','.join(self.exclude_file_list) + '\n')
        elif self.next_stage == 4:
            data.append(self.encode_done)
            fd.write(','.join(map(str, data)) + '\n')

    def finish_stage(self):
        self.next_stage += 1
        if self.next_stage > 4:
            self.next_stage = 1
            self.generation += 1
        self._save_log()
        return self.generation, self.next_stage

    def get_sample_count(self, dir_path):
        count = 0
        if not os.path.exists(dir_path):
            return 0
        for file in os.listdir(dir_path):
            full_path = os.path.join(dir_path, file)
            fd = open(full_path)
            header = fd.readline()
            for _ in fd:
                count += 1
        return count

    def is_encoded_data_ready(self):
        self._load_log()
        return self.encode_done

    def get_exclude_file_list(self):
        self._load_log()
        return self.exclude_file_list

    def add_exclude_file_list(self, file):
        self.exclude_file_list += file
        self._save_log()

    def finish_encode_data(self):
        self.encode_done = True
        self._save_log()


def main():
    ##################################################################
    # Arguments
    mcts_rl_data_home = '../data/rl_mcts/'
    progress_log_file = os.path.join(mcts_rl_data_home, 'progress_log.txt')

    encoder_name = 'fourplane'
    policy_cpu_threads = 1
    policy_use_gpu = False
    value_cpu_threads = 1
    value_use_gpu = False

    ##################################################################
    # Parameters
    policy_train_sample_count = int(800 * 1024 / 0.8 / 12)
    value_train_sample_count = int(200 * 1024 / 0.8 / 12)
    winrate_evaluation_game_count = 50
    winrate_evaluation_batch_size = 128
    policy_model_filename = 'policy_model.h5'
    value_model_filename = 'value_model.h5'

    data_per_file = 1024
    policy_epochs = 100
    value_epochs = 100

    ##################################################################
    # Load last state from a log file
    logger = RLMCTSTrainLogger(mcts_rl_data_home)
    generation = logger.generation
    stage = logger.next_stage

    prepare_tf_custom_objects()
    while True:
        ##################################################################
        # Prepare variables
        generation_home = os.path.join(mcts_rl_data_home, f'generation{generation:02d}')
        if generation == 0:
            prev_generation_home = None
            old_policy_model_path = None
            old_value_model_path = None
        else:
            prev_generation_home = os.path.join(mcts_rl_data_home, f'generation{generation-1:02d}')
            # Set models path
            old_policy_model_path = os.path.join(prev_generation_home, policy_model_filename)
            old_value_model_path = os.path.join(prev_generation_home, value_model_filename)
        new_policy_model_path = os.path.join(generation_home, policy_model_filename)
        new_value_model_path = os.path.join(generation_home, value_model_filename)

        # Set data directory path
        generated_game_dir = os.path.join(generation_home, 'generated_games')
        populated_games_dir = os.path.join(generation_home, 'populated_games')
        dataset_dir = os.path.join(generation_home, 'dataset')
        encoded_data_dir = os.path.join(generation_home, 'encoded_data')
        shuffled_data_dir = os.path.join(generation_home, 'shuffled_data')

        value_generated_game_dir = os.path.join(generation_home, 'value_generated_game')
        value_populated_game_dir = os.path.join(generation_home, 'value_populated_game')
        value_dataset_dir = os.path.join(generation_home, 'value_dataset_dir')

        print(f'======================= Generation {generation}: Stage {stage} =======================')
        if stage == 1:
            ##################################################################
            # Stage 1
            ##################################################################
            step_count = policy_train_sample_count - logger.get_sample_count(generated_game_dir)
            # Prepare directory
            Path(generated_game_dir).mkdir(parents=True, exist_ok=True)
            # Generate games
            encoder = get_encoder_by_name(encoder_name, 5, None, data_format="channels_last")
            cpu_threads = policy_cpu_threads
            use_gpu = policy_use_gpu
            gc.collect()
            print('Start to generate games for policy train')
            generate_games(generated_game_dir, encoder, old_policy_model_path, old_value_model_path, step_count,
                           cpu_threads, use_gpu, generation == 0)
        elif stage == 2:
            ##################################################################
            # Stage 2
            ##################################################################
            mode = 'policy'
            encoder = get_encoder_by_name(encoder_name, 5, mode, data_format="channels_last")
            if not logger.is_encoded_data_ready():
                # Prepare directory
                Path(populated_games_dir).mkdir(parents=True, exist_ok=True)
                Path(dataset_dir).mkdir(parents=True, exist_ok=True)

                # Preprocess data
                populator = GamePopulator(5)
                populated_games_name = os.path.join(populated_games_dir, 'game_%d.txt')
                dataset_name = os.path.join(dataset_dir, f'{data_per_file}_%06d.txt')
                populator.populate_games([generated_game_dir], populated_games_name, with_value=True)
                generate_dataset(populated_games_dir, data_per_file, dataset_name)

                try:
                    shutil.rmtree(encoded_data_dir)
                except FileNotFoundError:
                    pass
                Path(encoded_data_dir).mkdir(parents=True, exist_ok=True)
            # Prepare encoded data
            gc.collect()
            generator = DataGenerator(encoder, data_per_file, dataset_dir, encoded_data_dir, mode, 0.2)
            logger.finish_encode_data()

            # Train data
            gc.collect()
            model_generator = ac_simple1_v2.ACSimple1(mode, dropout_rate=0.3, data_format="channels_last")
            model = model_generator.model(encoder.shape(), encoder.num_moves(), optimizer=SGD(learning_rate=0.1))
            # If you want to train model from previous model, uncomment below line
            # model = load_model(old_policy_model_path)
            model.fit_generator(
                generator=generator.generate('train'),
                epochs=policy_epochs,
                steps_per_epoch=generator.get_num_steps('train'),
                validation_data=generator.generate('test'),
                validation_steps=generator.get_num_steps('test'),
                callbacks=[
                    ModelCheckpoint(new_policy_model_path)
                ]
            )
        elif stage == 3:
            ##################################################################
            # Stage 3
            ##################################################################
            Path(shuffled_data_dir).mkdir(parents=True, exist_ok=True)
            Path(value_generated_game_dir).mkdir(parents=True, exist_ok=True)
            encoder = get_encoder_by_name(encoder_name, 5, None, data_format="channels_last")
            # Shuffle data
            shuffled_data_name = os.path.join(shuffled_data_dir, f'shuffled_{data_per_file}_%06d.txt')
            generate_dataset(generated_game_dir, data_per_file, shuffled_data_name)

            file_batch_size = 1
            all_file_list = set(os.listdir(dataset_dir))
            all_file_list -= set(logger.get_exclude_file_list())
            all_file_list = list(all_file_list)
            while True:
                remain_samples = value_train_sample_count - logger.get_sample_count(value_generated_game_dir)
                if remain_samples == 0 or len(all_file_list) == 0:
                    break
                # Get file list to evaluate for this iteration
                input_list = []
                for f in all_file_list[:file_batch_size]:
                    full_path = os.path.join(dataset_dir, f)
                    input_list.append(full_path)
                logger.add_exclude_file_list(input_list)
                all_file_list = all_file_list[file_batch_size:]
                print(f'Evaluating {input_list}...')

                # Evaluate winrates
                use_gpu = value_use_gpu
                cpu_threads = value_cpu_threads
                gc.collect()
                evaluate_winrates(input_list, remain_samples, value_generated_game_dir, encoder, new_policy_model_path,
                                  winrate_evaluation_game_count, winrate_evaluation_batch_size, cpu_threads, use_gpu)
        elif stage == 4:
            ##################################################################
            # Stage 4
            ##################################################################
            mode = 'value'
            encoder = get_encoder_by_name(encoder_name, 5, mode, data_format="channels_last")
            if not logger.is_encoded_data_ready():
                # Prepare directory
                Path(value_populated_game_dir).mkdir(parents=True, exist_ok=True)
                Path(value_dataset_dir).mkdir(parents=True, exist_ok=True)

                # Preprocess data
                populator = GamePopulator(5)
                value_populated_games_name = os.path.join(value_populated_game_dir, 'game_%d.txt')
                value_dataset_name = os.path.join(value_dataset_dir, f'{data_per_file}_%06d.txt')
                populator.populate_games([value_generated_game_dir], value_populated_games_name, with_value=True)
                generate_dataset(value_populated_game_dir, data_per_file, value_dataset_name)

                try:
                    shutil.rmtree(encoded_data_dir)
                except FileNotFoundError:
                    pass
                Path(encoded_data_dir).mkdir(parents=True, exist_ok=True)
            # Prepare encoded data
            gc.collect()
            generator = DataGenerator(encoder, data_per_file, value_dataset_dir, encoded_data_dir, mode, 0.2)
            logger.finish_encode_data()

            # Train data
            model_generator = ac_simple1_v2.ACSimple1(mode, dropout_rate=0.3, data_format="channels_last")
            model = model_generator.model(encoder.shape(), encoder.num_moves(), optimizer=SGD(learning_rate=0.1))
            # If you want to train model from previous model, uncomment below line
            # model = load_model(old_value_model_path)
            gc.collect()
            model.fit_generator(
                generator=generator.generate('train'),
                epochs=value_epochs,
                steps_per_epoch=generator.get_num_steps('train'),
                validation_data=generator.generate('test'),
                validation_steps=generator.get_num_steps('test'),
                callbacks=[
                    ModelCheckpoint(new_value_model_path)
                ]
            )
        else:
            assert False, 'Invalid stage'
        generation, stage = logger.finish_stage()


if __name__ == '__main__':
    main()
