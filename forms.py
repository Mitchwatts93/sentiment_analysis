from wtforms import Form, TextField, validators, SubmitField
from flask import Flask, render_template, request
from utils import eval_from_input
from utils import load_model

#create app
app = Flask(__name__)

class WebForm(Form):
    eval_text = TextField("Please enter some text to be evaluated",
                          validators=[validators.InputRequired()])
    submit = SubmitField("Enter")

@app.route("/", methods=['GET', 'POST'])
def home():
    "home page with form"
    form = WebForm(request.form)
    model = load_model()


    if request.method == 'POST' and form.validate():
        eval_text = request.form['eval_text']
        return render_template('eval_text.html',
                               input=eval_from_input(eval_text=eval_text, model=model))

    return render_template('index.html', form=form)

if __name__ == "__main__":
    print(("* Loading model and Flask starting server..."
           "please wait until server has fully started"))
    # Run app
    app.run(host="0.0.0.0", port=80)

    # on local machine use 10000