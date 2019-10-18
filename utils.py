from visualisation import return_html

def eval_from_input(eval_text):
    """Generate output from a sequence"""

    ### replace this bit with generating an output instead
    html_result = generate_sentiment_html(eval_text)

    # Formatting in html
    html = ''
    html = addContent(html, header(
        'Sentiment Analysis', color='#FE9023'))
    html = addContent(html, box(html_result))
    return f'<div>{html}</div>'


def generate_sentiment_html(eval_text):
    sentiment, confidence, model = get_sentiment(eval_text)
    html = return_html(sentiment, confidence, model, eval_text)
    return html

def get_sentiment(eval_text):
    from flair.data import Sentence
    try:
        if model:
            pass
    except NameError as e:
        pass
    model = load_model()
    S = Sentence(eval_text)
    sentiment, confidence = eval_model(model, S)
    # should also do the explaining in here but will need to change name
    return sentiment, confidence, model  # this will be fixed in the future


def eval_model(model, sentence_obj):
    model.predict(sentence_obj)
    result = sentence_obj.labels[0]  # assuming theres only one sentence!
    sentiment, confidence = result.value, result.score
    return sentiment, confidence


def load_model():
    """Load in the pre-trained model"""
    from flair.models import TextClassifier
    global model
    model = TextClassifier.load('./en-sentiment')
    return model


def header(text, color='black'):
    """Create an HTML header"""
    raw_html = f'<h1 style="margin-top:12px;color: {color};font-size:54px"><center>' + str(
        text) + '</center></h1>'
    return raw_html


def box(text):
    """Create an HTML box of text"""
    raw_html = '<div style="border-bottom:1px inset black;border-top:1px inset black;padding:8px;font-size: 28px;">' + str(
        text) + '</div>'
    return raw_html


def addContent(old_html, raw_html):
    """Add html content together"""
    old_html += raw_html
    return old_html
