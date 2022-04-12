out_filename_format = ''
convert_dataset_to_npy(encoder=, batch=, in_dir=, out_filename_format=out_filename_format, test_ratio=, overwrite=False)
generator = DataGenerator(out_filename_format) # If you want to change batch, you have to re-convert data

## when call fit_generator
generator = generator.generator('train')
steps_per_epoch=generator.get_num_steps('train')
validation_data=generator.generator('test')
validation_steps=generator.get_num_steps('test')
