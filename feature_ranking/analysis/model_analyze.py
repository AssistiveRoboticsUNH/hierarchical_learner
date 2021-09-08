import pandas as pd
import numpy as np
import os

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=4)

def get_accuracy_c(args, df):

    if args.save_cm:
        dataset_path = input("Path to class labels: ")
        class_labels = sorted(os.listdir(dataset_path))

    for mode in ["train", "evaluation"]:
        print('----'+mode.upper()+'----')
        df_mode = df[df["mode"] == mode]

        expected = df_mode["expected_label"]
        predicted = df_mode["predicted_label"]

        norm = 'false' if args.normalize else 'true'
        cm = confusion_matrix(expected, predicted, normalize=norm)
        print(cm)

        accuracy = accuracy_score(y_true=expected, y_pred=predicted)
        print("accuracy:", accuracy)

        if args.save_cm:

            df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
            plt.figure(figsize=(10, 7))
            sns.set(font_scale=1.5)
            plot = sns.heatmap(df_cm, annot=True, cmap="YlGnBu", vmax=1)
            plt.tight_layout()

            plot.set_yticklabels(plot.get_yticklabels(), rotation=0)
            plot.figure.savefig(os.path.join(args.model, "cm_"+mode+".png"))


def parse_exec_args():
    import argparse
    parser = argparse.ArgumentParser(description='Execute file')

    parser.set_defaults(app='c')
    parser.add_argument('model', help='model_location')

    parser.set_defaults(save_cm=False)
    parser.add_argument('--save_cm', help='save confusion_matrix', dest='save_cm', action='store_true')
    parser.set_defaults(normalize=False)
    parser.add_argument('--no_norm', help='stop normalizing results', dest='normalize', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_exec_args()
    df = pd.read_csv(os.path.join(args.model, "results.csv"))

    get_accuracy_c(args, df)

