import argparse
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Resampling dataset args')

    parser.add_argument(
        '--path', dest='FILE_PATH',
        help='Path to file in jsonl format',
        type=str, required=True
    )

    parser.add_argument(
        '--type', dest='TYPE',
        choices=['oversample', 'undersample'],
        help='{oversample, undersample}',
        type=str, required=True
    )

    parser.add_argument(
        '--strategy', dest='STRATEGY',
        help='Strategy for under-/oversampling: float, str, dict, callable',
        type=any, required=False,
        default='all'
    )

    parser.add_argument(
        '--random_state', dest='RANDOM_STATE',
        help='random state',
        type=int, required=False, default=123
    )

    parser.add_argument(
        '--para-frame', dest='FRAME',
        help='Which framing is used: "Para-Frame1" or "Para-Frame2"',
        choices=[1, 2],
        type=int, required=True
    )
    args = parser.parse_args()
    return args


class Resampler():
    def __init__(self,
                 args):
        self.args = args

    def resample(self):
        df = pd.read_json(self.args.FILE_PATH, lines=True)
        X_tr = df["text"].values.reshape(-1, 1)
        y_tr = df["labels"]
        fr = self.args.FRAME
        if self.args.TYPE == 'oversample':
            print(f'Oversampling the dataset "{self.args.FILE_PATH}"...')
            oversampler = RandomOverSampler(sampling_strategy=self.args.STRATEGY,
                                            random_state=self.args.RANDOM_STATE)
            X_over, y_over = oversampler.fit_resample(X_tr, y_tr)
            df_over = pd.DataFrame({'text': X_over[:, 0], 'labels': y_over})
            df_over.to_json(f'oversampled-{fr}.jsonl', lines=True, orient='records')
            print(f'Success! \nNew file "oversampled-{fr}.jsonl" has been created.')

        else:
            print(f'Undersampling the dataset "{self.args.FILE_PATH}"...')
            undersampler = RandomUnderSampler(sampling_strategy=self.args.STRATEGY,
                                              random_state=self.args.RANDOM_STATE)
            X_under, y_under = undersampler.fit_resample(X_tr, y_tr)
            df_under = pd.DataFrame({'text': X_under[:, 0], 'labels': y_under})
            df_under.to_json(f'undersampled-{fr}.jsonl', lines=True, orient='records')
            print(f'Success! \nNew file "undersampled-{fr}.jsonl" has been created.')

if __name__ == '__main__':
    args = parse_args()
    sampler = Resampler(args)
    sampler.resample()
