# Introduction to Neural Network

## What is Deep Learning?
- Deep Learning is basically a field of study where in we use something known as neurons for traing the data.
- for example, we teach a baby how to identify the cat or dog or car.
- Automated Feature Engineering.
- In Machine Learning need data in feature space such as x1, x2, x3... 
- Where as Deep Learning No need of Featurization.
- Deep Learning can Extract the feature from Image, Text, Tabula data, Audio, Video, etc. it automatically extract information.
> **New Definition:** Deep Learning is basically a mathematical techinique that enable us to extract infromation from any kind of structured/unstructured data using the concept of neuron.
- $$ Y = f(x) $$ This data(x) can be Tabular/graph/video/Audio/Text/Image etc; we can give any kind of data and we can do any kind of prediction. the only kind we need to understand which architure of f or which deep learning techinique we should use in this case, this is the power of Deep Learning.
- if we design is arthitecure of f in proper manner, we can solve any kind of problem.

- Q. You are saying that data can be anything, but you also saying that function the deep learning algorithm should be articureize according to the data, can you tell what type of data vs what type of archtecure works.
- ans. that is why different deep learning techinques exist.

## Different Deep Learning Techinques.
1. **Artifical Neural Network** - This type of neural network are design to extract infomation from tabular data. like csv file.
2. **Convolution Neural Network** - This type of neural network are design to extract infomation from image data.
3. **Recurent Neural Network(LSTM/GRU/Transformers)** - This type of neural network are design to extract information from squence data(like text, audio, Videos).
    - RNN has many class model such as LSTM(long short term memeory/GRU)
    - Tranformers are like a part of RNN for Squence data. ex - chatGPT
4. **Graph Neural Network** - for Graphs(molecule)

### Multi-Modality: 
- Multi-Modality is a concept where single functioncan take data from multiple sources, I can take tablular data, image data, text data together, etc. not the enemble or average out of model, together mean join of all the data.
- All the infromation from multiple possible resouces, just like human brain, our brain is desion multi-modal. 
- for example; To understand anything I can use visual, i can see the object, I can smell etc.
- taking infomation from multiple sources.
- think of medical data; test, X-rays, ECG(time series data) that is what multi-modelity. make decision depended on the differnt join input.

#### important points
- take text --> generate an image(stable deffusion model)
- take image --> generate the text (captioning model)
- take text --> and generate a video(Runway company doing that)

### Why Deep Learning vs Machine Learning?
- for example: predicting secand hand car24, you are not only using tablular data but also image of car, inspection command, service history, or other comments. to pridict the price of second hand car. which machine learning can not do, this is the beauti of deep learning.

#### Desision Tree also partiting the data neural network also partiting the data it is like a black box, but we know that how decision tree partiting, verticle lines and horizontal line, but we don't know, how neural network is partiting the data, it is partiting in small small peaces and find the curve. (but there are way to tell us)

### Why the non-linearlity is absolutely vital?

<p align="center">
  <img src="https://learnopencv.com/wp-content/uploads/2017/10/mlp-diagram.jpg" alt="Logo" style="width: 80%;">
</p>

$$
 n_1 = x_1w_1 + x_2w_2 + wo 
$$

$$
 n_2 = x_1w_3 + x_2w_4 + wo
$$

$$
    w_1, w_2, w_3, w_4 --> these thing decide my slope, and when it pass to the non-linearity it will decide my curvature
$$

$$
 y^{ \wedge } = w_5n_1 + w_6n_2 + wb
$$

$$
 y^{ \wedge } = w_5(x_1w_1 + x_2w_2 + wo) + w_6(x_1w_3 + x_2w_4 + wo) + wb
$$

$$
 y^{ \wedge } = w_5x_1w_1 + w_5x_2w_2 + w_5wo + w_6x_1w_3 + w_6x_2w_4 + w_6wo + wb
$$

- collect all the x1, x2, wb

$$
 y^{ \wedge } = x_1(w_5w_1 + w_6w_3) + x_2(w_5w_2 + w_6w_4) + w_5wo  + w_6wo + wb
$$

- it's a line:

$$
 y^{ \wedge } = x_1A + x_2B + C
$$

- if we add any kind of non linearity then the megic will happen.
- this is the beauti, 1 neuron doing the simple task but when you join simple, simple, ... task to data i will become big and solve bigger problem.

$$
 n_1 = \tanh\left(x_1w_1 + x_2w_2 + wo\right)
$$

$$
 n_2 = \tanh\left(x_1w_3 + x_2w_4 + wo\right)
$$

- That is a single curvatue of the a line can fit any decision boundary

## How do we train such a network of neurons:
1. Intilise the architecure (define the number of hidden layers, neurons etc)
1. Randomly Intilise the weight
1. Given the weights calculate the loss:-
    * how do i calculate the loss of regression problem? 
        MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
        Where:
        - n is the number of observations,
        - yᵢ is the actual value for the i-th observation,
        - ŷᵢ is the predicted value for the i-th observation.
1. Given the loss, update the weithts- Gradient Descent Algorithm + Chain Rule
1. using the updated weights go to step3 and keep on doing it till you stablise your training loss to the min possible(min val loss)

# Gradient Descent Algothim and Chain Rule
- Gradient will have some problems and that where we will be learning optimizers; it is meant to improve the gradient descent problem.




