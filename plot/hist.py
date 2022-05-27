import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_histogram():
    df = pd.DataFrame(
        {
            "category": [
                "1-shot 5-way",
                "5-shot 5-way",
                "1-shot 20-way",
                "5-shot 20-way",
                "1-shot 5-way",
                "5-shot 5-way",
                "1-shot 20-way",
                "5-shot 20-way",
            ],
            "legend": [
                "our",
                "our",
                "our",
                "our",
                "article",
                "article",
                "article",
                "article",
            ],
            "data": [96.48, 98.92, 93.22, 98.29, 99.8, 99.7, 96.0, 98.9],
        }
    )

    sns.catplot(data=df, kind="bar", x="category", y="data", hue="legend")
    plt.show()


if __name__ == "__main__":
    make_histogram()
