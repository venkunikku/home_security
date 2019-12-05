# Personalized Security Camera

## To get possible commands:
```angular2html
python3 face_detection_and_embedings.py --help
```

## To Run a specific model - without creating the train data embeddings:
```
python3 face_detection_and_embedings.py --cam True  --create_embeddings False --train_model True --face_recognize True --train_data ../Data/image_data_set/ --ml_model LDA  
python3 face_detection_and_embedings.py --cam True  --create_embeddings False --train_model True --face_recognize True --train_data ../Data/image_data_set/ --ml_model GBC  
python3 face_detection_and_embedings.py --cam True  --create_embeddings False --train_model True --face_recognize True --train_data ../Data/image_data_set/ --ml_model RF
python3 face_detection_and_embedings.py --cam True  --create_embeddings False --train_model True --face_recognize True --train_data ../Data/image_data_set/ --ml_model SVC
```
## Our best performing model:
```
python3 face_detection_and_embedings.py --cam True  --create_embeddings False --train_model True --face_recognize True --train_data ../Data/image_data_set/ --ml_model SVC
```
## To Run a specific model - with creating the train data embeddings:
### This will train again with all the data and creates the embeddings again to by used.
```
python3 face_detection_and_embedings.py --cam True  --create_embeddings True --train_model True --face_recognize True --train_data ../Data/image_data_set/ --ml_model SVC
```

## To run the model with testing data (change the model to run with respective model):
```
python3 face_detection_and_embedings.py --test_data_path ../Data/image_data_set_test --test True --ml_model SVC

```

## If you want to just start your camera and do inference without creating the embedding and train the embeddings:
```angular2html
python3 face_detection_and_embedings.py --cam True --face_recognize True --ml_model SVC --record False  
```

## If you want to start live camera and also record:
```angular2html
python3 face_detection_and_embedings.py --cam True --face_recognize True --ml_model SVC --record True
```

## If you want to create your own images are test data:
```angular2html
python3 face_dataset_creator.py --help  
python3 face_dataset_creator.py  --out ../Data/image_data_set/venku -fn=120
```