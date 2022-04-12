import os
import multiprocessing
import queue
import time


def encode_data_worker(encoder, q_infile, q_outfile):
    print('4323243214')
    while True:
        try:
            q_outfile.put(q_infile.get_nowait())
        except queue.Empty:
            break
    print('end worker!')


def convert_dataset_to_npy(encoder, batch, in_dir, out_filename_format, test_ratio, overwrite=False):
    # Count file
    data_files = [f for f in map(lambda x: os.path.join(in_dir, x), os.listdir(in_dir)) if os.path.isfile(f)]
    file_count = len(data_files)
    train_file_count = int(file_count * (1 - test_ratio))

    # Separate file for train and test
    train_files = data_files[:train_file_count]
    test_files = data_files[train_file_count:]
    # train_files = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # Put file names & Make threads to encode data
    q_infile = multiprocessing.Queue()
    q_outfile = multiprocessing.Queue()

    for file in train_files:
        q_infile.put(file)

    thread_count = os.cpu_count() - 1
    thread_list = []
    for _ in range(thread_count):
        p = multiprocessing.Process(target=encode_data_worker, args=(encoder, q_infile, q_outfile))
        p.start()
        thread_list.append(p)

    # Get encoded data and do work
    all_thread_end = False
    while not all_thread_end:
        # If all encoder worker finished, exit this loop after processing remain things
        all_thread_end = True
        for p in thread_list:
            if p.is_alive():
                all_thread_end = False
                break

        while not q_outfile.empty():
            # Do work:
            print(q_outfile.get())
        time.sleep(0.01)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    convert_dataset_to_npy(0,0,0,0,0)
