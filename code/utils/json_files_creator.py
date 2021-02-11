import json
import glob, os
import random


class json_store_path(path_to_explore,path_to_store,file_name, file_extension='.jpg'):
    def __init__():
        self.path_to_explore=path_to_explore
        self.path_to_store=path_to_store
        self.json_name=file_name
        self.file_extension=file_extension

    def json_generator(self, split=False):
        files=[]
        for file in os.listdir(full_path_to_explore):
            if file.endswith(file_extension_to_use):
                files.append(os.path.join(full_path_to_explore, file))

        if split==True:
            files_training=[]
            files_validation=[]

            #
            number_item_train_set=int(len(files)*(1-(validation_percentage/100)))
            train_index=[random.randint(0,len(files)) for iter in range(number_item_train_set)]

            for i in range(len(files)):
                if i in train_index:
                    files_training.append(files[i])
                else:
                    files_validation.append(files[i])

            with open(path_to_store_json+'/'+name_json_file+'.json', 'w') as f:
                    json.dump(files_training, f)

            with open(path_to_store_json+'/'+name_json_file+'_val_'+'.json', 'w') as f:
                    json.dump(files_validation, f)

        else:

            with open(path_to_store_json+'/'+name_json_file+'.json', 'w') as f:
                    json.dump(files, f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'introduce dataset folder')
	parser.add_argument('-p', metavar='dataset_directory', required=True,
	                help='the path to the directory containing the Json file')

	parser.add_argument('-o', metavar='output directory', required=True,
	                help='')

	parser.add_argument('-n', metavar='name of the output', required=True,
	                help='the path to the directory containing the Json file')

	parser.add_argument('-e', metavar='file extentension', required=False,
	                help='the path to the directory containing the Json file')

    args = parser.parse_args()

    if len(args.e)>1:
        json_store_path(args.p,args.o,args.n, file_extension=args.e)

    else:
        json_store_path(args.p,args.o,args.n)
