% Prepare train, test and val files

clear
clc

fidt = fopen('train.txt', 'w');
fidv = fopen('val.txt', 'w');
fidtest = fopen('test.txt', 'w');

main_path = '/your/dataset/folder/path';
sets = dir(main_path);
for k=3:length(sets)
	full_path = [main_path '/' sets(k).name];
	folders = dir(full_path);

	for i=3:length(folders)
		images = dir([full_path '/' folders(i).name]);
		for j=3:length(images)
			if strcmp(sets(k).name, 'train')
				fprintf(fidt, '%s ', [full_path '/' folders(i).name '/' images(j).name]);
				fprintf(fidt, '%s', num2str(i-3));
				fprintf(fidt, '\n');
			elseif strcmp(sets(k).name, 'test')
				fprintf(fidtest, '%s ', [full_path '/' folders(i).name '/' images(j).name]);
				fprintf(fidtest, '%s', num2str(i-3));
				fprintf(fidtest, '\n');
			else
				fprintf(fidv, '%s ', [full_path '/' folders(i).name '/' images(j).name]);
				fprintf(fidv, '%s', num2str(i-3));
				fprintf(fidv, '\n');
			end
		end
		disp(folders(i).name)	
	end
end

fclose(fidt);
fclose(fidv);
fclose(fidtest);