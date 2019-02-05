c:

cd c:\caffe

build\tools\Release\convert_imageset E:\AFLW\data\ E:\AFLW\aflw-matlab\Nam\train256.txt  E:\AFLW\data\lmdb3\train_lmdb

build\tools\Release\convert_imageset E:\AFLW\data\ E:\AFLW\aflw-matlab\Nam\valid256.txt  E:\AFLW\data\lmdb3\valid_lmdb

build\tools\Release\compute_image_mean E:\AFLW\data\lmdb3\train_lmdb E:\AFLW\data\mean\mean_image3.binaryproto

pause