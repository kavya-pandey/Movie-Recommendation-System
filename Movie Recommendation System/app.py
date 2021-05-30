#%% 
 
from flask import Flask, request, render_template
#render_template helps us ro redirect to the home page that we have initially then in that home page we will try to pt some input 

app = Flask(__name__, template_folder='template')  #__name__ is a special variable that gets as value the string "__main__" when you’re executing the script.

import recommendation

@app.route("/")    #root node, root api url where it should go directly directed to index.html file we have use'/' in flas to create any numbers of uri(s) w.r.t. api(s) the function is mapped to /(home) url

def home():
    return render_template('index.html')
#the api 
@app.route('/prediction',methods=['POST'])
def prediction():
    
    features = request.form.get("movie")
    movies = recommendation.movie_recommendation_function(features)
    return render_template('index.html', prediction_text=movies) #prediction_test should be replaced with movies
if __name__ == "__main__": #Python assigns the name "__main__" to the script when the script is executed.
    app.run(debug=True)
#%%
