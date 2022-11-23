# DLHLP-Prosody turn taking

## How to use

### preprocessing

```
python preprocess_wav.py --input_path /path/to/maptask/data
python preprocess_timed_units.py --input_path /path/to/maptask/data
python get_split_table --input_path /path/to/maptask/data
```

### set up environment (TWCC)

```
conda env update --name <name> --file environment.yml 
```

### Data after preprocessing

```
Data_out/
    dialogues-splitted/
	q1nc1-0.mix.wav
	q1nc1-1.mix.wav
	q1nc1-2.mix.wav
		...
    metadata/
	q1ec1.f.csv
	q1ec1.g.csv
	q1ec2.f.csv
	q1ec2.g.csv
		...
    split-tables/
	train_ids.csv
	dev_ids.csv
	test_ids.csv
    timed-units/
	q1ec1.f.csv
	q1ec1.g.csv
	q1ec2.f.csv
	q1ec2.g.csv
		...
```

- dialogues-splitted
  - split dialogues to 60 seconds

- metadata
  - dialogues metadata

- split-tables
  - split data into train, dev, test

- timed-units
  - dialogues timed-units (50ms) 

### Train

```
EXP_NAME=DLHLP_1
UPSTREAM=fbank
DOWNSTREAM=turn_taking

python3 run_downstream.py -f -l -1 -m train -n $EXP_NAME -u $UPSTREAM -d $DOWNSTREAM
```
