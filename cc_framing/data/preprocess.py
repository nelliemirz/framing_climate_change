import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Preprocess args')

    parser.add_argument(
        '--path', dest='FILE_PATH',
        help='Path to file in jsonl format',
        type=str, required=True
    )

    parser.add_argument(
        '--para-frame', dest='FRAME',
        help='Which framing to keep: "Para-Frame1" or "Para-Frame2"',
        choices=[1, 2],
        type=int, required=True
    )

    parser.add_argument(
        '--split', dest='SPLIT',
        help='Split dataset into train/test/val splits',
        action='store_true'
    )

    parser.add_argument(
        '--val', dest='VAL',
        help='Whether to split data into train/test/val, otherwise data is split into train/test',
        action='store_true'
    )

    parser.add_argument(
        '--drop-rare', dest='DROP_RARE',
        help='Whether to drop labels with rare counts, minimal number can be specified with --min_size argument',
        action='store_true'
    )

    parser.add_argument(
        '--min-size', dest='MIN_SIZE',
        help='The minimum number of times a label should appear in the dataset',
        type=int, required=False, default=20
    )

    parser.add_argument(
        '--top3-split', dest='TOP3',
        help='Whether to split dataset into two parts: top 3 labels and the rest',
        action='store_true'
    )

    parser.add_argument(
        '--test-size', dest='TEST_SIZE',
        help='Test size',
        type=float, required=False, default=0.2
    )

    parser.add_argument(
        '--no-prep', dest='NO_PREP',
        help='Do not preprocess the file',
        action='store_true'
    )

    args = parser.parse_args()
    return args


class Preprocess(object):
    def __init__(self, args):
        """
        Params:
              file_path - Input Dataframe to be pre-processed
        """
        self.args = args

    def preprocess(self):
        print('Cleaning dataset...')
        fr = self.args.FRAME
        os.makedirs(f'./preprocessed{fr}', exist_ok=True)
        df = pd.read_csv(self.args.FILE_PATH, sep='\t')
        if self.args.FRAME == 1:
            df = df[['Text', 'Para-Frame1']]
            df.dropna(axis=0, inplace=True)
            df = df[~df['Para-Frame1'].isin(['-', '%', np.nan])]
            df = df.rename(columns={'Text': 'text', 'Para-Frame1': 'labels'})
        else:
            df = df[['Text', 'Para-Frame2']]
            df.dropna(axis=0, inplace=True)
            df = df[~df['Para-Frame2'].isin(['-', '%', np.nan])]
            df = df.rename(columns={'Text': 'text', 'Para-Frame2': 'labels'})
        return df

    def split_data(self, df):
        fr = self.args.FRAME
        print('Splitting into sets...')
        train, devtest = train_test_split(df, test_size=self.args.TEST_SIZE, shuffle=True, random_state=42)
        if self.args.VAL:
            dev, test = train_test_split(devtest, test_size=0.50, shuffle=True, random_state=42)
            train.to_json(f'./preprocessed{fr}/train.jsonl', lines=True, orient='records')
            dev.to_json(f'./preprocessed{fr}/val.jsonl', lines=True, orient='records')
            test.to_json(f'./preprocessed{fr}/test.jsonl', lines=True, orient='records')
            print('Train/Val/Test splits saved to "preprocessed" directory')
            return train, dev, test
        else:
            train.to_json(f'./preprocessed{fr}/TRAIN.jsonl', lines=True, orient='records')
            devtest.to_json(f'./preprocessed{fr}/TEST.jsonl', lines=True, orient='records')
            print(f'Train/Test splits saved to "preprocessed{fr}" directory')
            return train, devtest

    def drop_rare(self, df):
        fr = self.args.FRAME
        print(f'Dropping labels occuring less than {self.args.MIN_SIZE} times...')
        df = df.groupby('labels').filter(lambda x: len(x) > self.args.MIN_SIZE)
        df.to_json(f'./preprocessed{fr}/preprocessed_drop_rare.jsonl', lines=True, orient='records')

        return df

    def top_rest_split(self, df):
        print('Splitting into top 3 and rest sets...')
        dcf = df.groupby(['labels'], as_index=False).count()
        top_3 = dcf.nlargest(3, 'text').labels.to_list()
        top_3_df = df.loc[df['labels'].isin(top_3)]
        rest_df = df[~df['labels'].isin(top_3)]
        return top_3_df, rest_df

    def run(self):
        fr = self.args.FRAME

        if self.args.NO_PREP:
            df = pd.read_json(self.args.FILE_PATH, lines=True)
        else:
            df = self.preprocess()
            df.to_json(f'./preprocessed{fr}/preprocessed_dataset.jsonl', lines=True, orient='records')
            print(f'Preprocessed data saved to "./preprocessed{fr}/preprocessed_dataset.jsonl"')

        if self.args.SPLIT and self.args.DROP_RARE:
            data = self.drop_rare(df)
            self.split_data(data)


        elif self.args.DROP_RARE and self.args.TOP3:
            data = self.drop_rare(df)
            os.makedirs(f'./top3_{fr}', exist_ok=True)
            top, rest = self.top_rest_split(data)
            top.to_json(f'./top3_{fr}/top_3_drop_rare.jsonl', lines=True, orient='records')
            rest.to_json(f'./top3_{fr}/rest_drop_rare.jsonl', lines=True, orient='records')
            print(f'Dropped rare labels, split into Top/Rest. \nFiles saved to "top3_{fr}" directory')

        elif self.args.SPLIT:
            self.split_data(df)

        elif self.args.DROP_RARE:
            os.makedirs(f'./preprocessed{fr}', exist_ok=True)
            data = self.drop_rare(df)
            data.to_json(f'./preprocessed{fr}/df_no_rare.jsonl', lines=True, orient='records')
            print(f'Dropped rare labels, file saved to "./drop_rare_{fr}" directory')

        elif self.args.TOP3:
            os.makedirs(f'./top3_{fr}', exist_ok=True)
            top, rest = self.top_rest_split(df)
            top.to_json(f'./top3_{fr}/top_3_labels.jsonl', lines=True, orient='records')
            rest.to_json(f'./top3_{fr}/rest_labels.jsonl', lines=True, orient='records')
            print(f'Split into Top/Rest. \nFiles saved to "top3_{fr}" directory')


if __name__ == "__main__":
    args = parse_args()
    pre = Preprocess(args)
    pre.run()


