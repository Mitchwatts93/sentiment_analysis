from flair.models import TextClassifier
from flair.data import Sentence
from tqdm import tqdm as tqdm
import numpy as np
from functools import partial
from contextlib import contextmanager
from multiprocessing import Pool
from multiprocessing import cpu_count

@contextmanager
def poolcontext(*args, **kwargs):

    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def predictor(text):
    prediction = predict(text, multi_class_prob = True)
    return prediction


#### the lime explainer doesn't work so well for positive sentiment, it gets it correct overall but maybe something weird going on?

class FlairExplainer:

    def __init__(self, clf):
        self.classifier = clf

    def predict(self, texts):
        docs = list([Sentence(text) for text in texts])
        global predict
        print('made predictor global')
        predict = self.classifier.predict
        print('made func global')
        with poolcontext(processes=cpu_count()) as pool:
            print('running')
            docs[:] = list(pool.imap(predictor, docs))
        labels = [[x.value for x in doc[0].labels] for doc in docs]#assumes only one sentence per doc
        probs = [[x.score for x in doc[0].labels] for doc in docs]
        probs = np.array(probs)   # Convert probabilities to Numpy array

        # For each prediction, sort the probability scores in the same order for all texts
        result = []
        for label, prob in zip(labels, probs):
            order = np.argsort(np.array(label))
            result.append(prob[order])
        return np.array(result)


def explainer(clf, text):
    """Run LIME explainer on provided classifier"""
    from lime.lime_text import LimeTextExplainer

    model = FlairExplainer(clf)
    predictor = model.predict

    explainer = LimeTextExplainer(
        split_expression=lambda x: x.split(),
        bow=False,
        class_names=["NEGATIVE", "POSITIVE"]
    )
    exp = explainer.explain_instance(
        text,
        classifier_fn=predictor,
        num_samples=1000
    )
    return exp

def visualise_sentiments(data):
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import seaborn as sns
    import pandas as pd
    cmap = cm.coolwarm.reversed()
    fig = plt.gcf()
    figsize = fig.get_size_inches()
    fig.set_size_inches(figsize[0]*3, figsize[1]*2) # make it wider than default - a bit of hacky solution
    ax = sns.heatmap(pd.DataFrame(data).set_index("Sentence").T, center=0, annot=True, cmap=cmap)
    fig = ax.get_figure()
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.3)
    plt.tight_layout()
    del ax
    return fig

def return_html(sentiment, confidence, model, eval_text):
    sentiment = (-1)**(sentiment=="NEGATIVE") # a bit hacky so sort that out. is 1 if positive and -1 if neg

    exp = explainer(model, eval_text)
    keys = [key[0] for key in exp.as_map()[1]] # get the ranked order of the keys
    zipped_words = zip(list(exp.as_list()), keys) # get a list of tuples ((word, score), key)
    scores = [x for x, _ in sorted(zipped_words, key=lambda pair: pair[1])] # sort by the key so now in original order

    import matplotlib.pyplot as plt
    fig = visualise_sentiments({
        "Sentence": ["SENTENCE"] + [score[0] for score in scores],
        "Sentiment": [confidence * sentiment] + [score[1] for score in scores],
    })
    html = html_from_fig(fig)
    plt.clf()
    return html

def html_from_fig(fig):
    from io import BytesIO
    import base64
    buf = BytesIO()
    fig.savefig(buf, format="png", )
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii") # see img scaling setup for html?
    return f"<img src='data:image/png;base64,{data}'; style='height: 100%; width: 100%; object-fit: contain'/>" # f"<img src='data:image/png;base64,{data}'/>"