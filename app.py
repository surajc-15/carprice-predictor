from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
import pickle
import warnings

#creating an flask app
app=Flask(__name__)


from sklearn.exceptions import InconsistentVersionWarning
warnings.simplefilter("error", InconsistentVersionWarning)

app.static_folder = 'static'

model=pickle.load(open("models/xgboost.pkl",'rb'))



database={'Maruti Wagon': 124,
  'Hyundai Creta': 65,
  'Honda Jazz': 60,
  'Maruti Ertiga': 112,
  'Audi A4': 2,
  'Hyundai EON': 66,
  'Nissan Micra': 154,
  'Toyota Innova': 197,
  'Volkswagen Vento': 207,
  'Tata Indica': 180,
  'Maruti Ciaz': 109,
  'Honda City': 58,
  'Maruti Swift': 121,
  'Land Rover': 87,
  'Mitsubishi Pajero': 152,
  'Honda Amaze': 53,
  'Renault Duster': 164,
  'Mercedes-Benz New': 138,
  'BMW 3': 12,
  'Maruti S': 118,
  'Audi A6': 3,
  'Hyundai i20': 78,
  'Maruti Alto': 106,
  'Honda WRV': 63,
  'Toyota Corolla': 194,
  'Mahindra Ssangyong': 96,
  'Maruti Vitara': 123,
  'Mahindra KUV': 90,
  'Mercedes-Benz M-Class': 137,
  'Volkswagen Polo': 205,
  'Tata Nano': 183,
  'Hyundai Elantra': 67,
  'Hyundai Xcent': 76,
  'Mahindra Thar': 98,
  'Hyundai Grand': 70,
  'Renault KWID': 166,
  'Hyundai i10': 77,
  'Nissan X-Trail': 158,
  'Maruti Zen': 125,
  'Ford Figo': 47,
  'Mercedes-Benz C-Class': 128,
  'Porsche Cayenne': 160,
  'Mahindra XUV500': 101,
  'Nissan Terrano': 157,
  'Honda Brio': 56,
  'Ford Fiesta': 46,
  'Hyundai Santro': 72,
  'Tata Zest': 192,
  'Maruti Ritz': 117,
  'BMW 5': 13,
  'Toyota Fortuner': 196,
  'Ford Ecosport': 44,
  'Hyundai Verna': 75,
  'Datsun GO': 31,
  'Maruti Omni': 116,
  'Toyota Etios': 195,
  'Jaguar XF': 83,
  'Maruti Eeco': 111,
  'Honda Civic': 59,
  'Volvo V40': 210,
  'Mercedes-Benz B': 127,
  'Mahindra Scorpio': 95,
  'Honda CR-V': 57,
  'Mercedes-Benz SLC': 143,
  'BMW 1': 11,
  'Chevrolet Beat': 23,
  'Skoda Rapid': 174,
  'Audi RS5': 9,
  'Mercedes-Benz S': 140,
  'Skoda Superb': 175,
  'BMW X5': 18,
  'Mercedes-Benz GLC': 134,
  'Mini Countryman': 147,
  'Chevrolet Optra': 27,
  'Renault Lodgy': 168,
  'Mercedes-Benz E-Class': 131,
  'Maruti Baleno': 107,
  'Skoda Laura': 172,
  'Mahindra NuvoSport': 92,
  'Skoda Fabia': 171,
  'Tata Indigo': 181,
  'Audi Q3': 6,
  'Skoda Octavia': 173,
  'Audi A8': 5,
  'Mahindra Verito': 99,
  'Mini Cooper': 146,
  'Hyundai Santa': 71,
  'BMW X1': 16,
  'Hyundai Accent': 64,
  'Hyundai Tucson': 74,
  'Mercedes-Benz GLE': 135,
  'Maruti A-Star': 105,
  'Fiat Grande': 35,
  'BMW X3': 17,
  'Ford EcoSport': 43,
  'Audi Q7': 8,
  'Volkswagen Jetta': 203,
  'Mercedes-Benz GLA': 133,
  'Maruti Celerio': 108,
  'Tata Sumo': 187,
  'Honda Accord': 52,
  'BMW 6': 14,
  'Tata Manza': 182,
  'Chevrolet Spark': 29,
  'Mini Clubman': 145,
  'Nissan Teana': 156,
  'Maruti 800': 104,
  'Honda BRV': 55,
  'Jaguar XE': 82,
  'Tata Xenon': 191,
  'Audi A3': 1,
  'Mercedes-Benz GL-Class': 132,
  'Honda BR-V': 54,
  'Volvo S80': 209,
  'Renault Captur': 163,
  'Chevrolet Enjoy': 26,
  'Mahindra Bolero': 88,
  'Audi Q5': 7,
  'Mitsubishi Cedia': 148,
  'Maruti S-Cross': 119,
  'Skoda Yeti': 176,
  'Ford Endeavour': 45,
  'Mercedes-Benz GLS': 136,
  'Mercedes-Benz A': 126,
  'Maruti SX4': 120,
  'Toyota Camry': 193,
  'Honda Mobilio': 61,
  'Fiat Linea': 36,
  'Audi TT': 10,
  'Mahindra Renault': 94,
  'Jeep Compass': 85,
  'Ford Ikon': 50,
  'Chevrolet Sail': 28,
  'Mahindra Quanto': 93,
  'Chevrolet Aveo': 22,
  'Mahindra Xylo': 102,
  'Maruti Esteem': 113,
  'Tata Safari': 186,
  'Maruti Ignis': 115,
  'Jaguar XJ': 84,
  'Nissan Sunny': 155,
  'Mercedes-Benz SLK-Class': 144,
  'Volkswagen Passat': 204,
  'Maruti Dzire': 110,
  'Chevrolet Cruze': 25,
  'Renault Koleos': 167,
  'Toyota Qualis': 199,
  'Volkswagen Ameo': 200,
  'Maruti Grand': 114,
  'Datsun redi-GO': 33,
  'Smart Fortwo': 177,
  'Mitsubishi Outlander': 151,
  'Porsche Cayman': 161,
  'Mercedes-Benz CLA': 129,
  'Volvo XC60': 211,
  'Tata New': 184,
  'Porsche Boxster': 159,
  'Mahindra XUV300': 100,
  'Tata Hexa': 179,
  'Tata Tiago': 188,
  'BMW 7': 15,
  'Fiat Avventura': 34,
  'Tata Tigor': 189,
  'Volvo S60': 208,
  'Ambassador Classic': 0,
  'Volkswagen Beetle': 201,
  'Fiat Petra': 37,
  'Hyundai Getz': 69,
  'Audi A7': 4,
  'Hyundai Elite': 68,
  'Ford Aspire': 41,
  'Volkswagen Tiguan': 206,
  'Chevrolet Captiva': 24,
  'Fiat Punto': 38,
  'Mahindra TUV': 97,
  'BMW X6': 19,
  'Tata Bolt': 178,
  'Nissan Evalia': 153,
  'Renault Scala': 170,
  'Mahindra Jeep': 89,
  'Hyundai Sonata': 73,
  'Ford Freestyle': 48,
  'Mahindra Logan': 91,
  'Chevrolet Tavera': 30,
  'Volvo XC90': 212,
  'Renault Pulse': 169,
  'Mitsubishi Montero': 150,
  'Porsche Panamera': 162,
  'Volkswagen CrossPolo': 202,
  'Renault Fluence': 165,
  'Tata Venture': 190,
  'Tata Nexon': 185,
  'Isuzu MUX': 80,
  'Toyota Platinum': 198,
  'Mercedes-Benz R-Class': 139,
  'Mercedes-Benz CLS-Class': 130,
  'ISUZU D-MAX': 79,
  'Mercedes-Benz S-Class': 141,
  'Mitsubishi Lancer': 149,
  'Ford Classic': 42,
  'Datsun Redi': 32,
  'Ford Mustang': 51,
  'Ford Fusion': 49,
  'Fiat Siena': 39,
  'Maruti 1000': 103,
  'Mercedes-Benz SL-Class': 142,
  'BMW Z4': 20,
  'Force One': 40,
  'Maruti Versa': 122,
  'Honda WR-V': 62,
  'Bentley Continental': 21,
  'Lamborghini Gallardo': 86,
  'Jaguar F': 81}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    #you need to get all the values
    try:
        carmodel = request.form.get('carmodel')
        location=request.form.get("location")
        year=request.form.get("year")
        kilometer=eval(request.form.get("kilometer"))
        fueltype=request.form.get("fueltype")
        transmission=request.form.get("transmission")
        ownertype=request.form.get("ownertype")
        mileage=eval(request.form.get("mileage"))
        engine=eval(request.form.get("engine"))
        Power=eval(request.form.get("Power"))
        seat=request.form.get("seat")
    #now map the car model from dataset
        feature=[carmodel,location,year,kilometer,fueltype,transmission,ownertype,mileage,engine,Power,seat]
        feature1= ['carmodel', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission',
            'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats']
        data=pd.DataFrame(columns=feature1)
        datase=np.array(feature)
        data= data._append(pd.Series({'carmodel':carmodel,"Location":location,"Year":year,'Kilometers_Driven':kilometer,'Fuel_Type':fueltype,'Transmission':transmission,'Owner_Type':ownertype,'Mileage':mileage,'Engine':engine,'Power':Power,'Seats':seat}),ignore_index=True )
        prediction=model.predict(data)[0]
        price=np.exp(prediction)
        pricevalue="The car price is {} lakhs".format(price)
        return render_template("price.html",prediction_text=pricevalue)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__=="__main__":
    app.run(debug=True)
