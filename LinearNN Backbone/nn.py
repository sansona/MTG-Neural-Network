import numpy as np
from bokeh.plotting import figure, output_file, show
from Weights import initialize, update_weights


#activation functions
def sigmoid(x, deriv=False):
    if (deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

def tanh(x, deriv=False):
    if (deriv==True):
        return 4/((np.exp(-x) + np.exp(x))**2)

    return (1 - np.exp(-2*x))/(1 + np.exp(-2*x))


X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1],
            [1,1,0]])

y = np.array([[0],
            [1],
            [1],
            [0],
            [1]])

#Initialize lists for plotting cost function 
error = []
num_iter = []

#np.random.seed(1)

#if first run/reset weights
W0, W1  = initialize()

#if continue using weights from previous iterations
#W0 = np.loadtxt('W0.txt').reshape((3, 4))
#W1 = np.loadtxt('W1.txt').reshape((4, 1))

for i in range(600):

    l0 = X
    l1 = tanh(np.dot(l0, W0))
    l2 = sigmoid(np.dot(l1, W1))

    l2_error = y - l2
    
    #for plotting
    error.append(np.mean(np.abs(l2_error)))
    num_iter.append(i)
    
    #periodically updates with error message while iterating
    if (i % 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l2_error))))

    #gradient for layer 2 (output layer)
    l2D = l2_error*sigmoid(l2, deriv=True)
    
    #gradient for layer 1 (hidden layer)
    l1_error = l2D.dot(W1.T)
    l1D = l1_error * tanh(l1, deriv=True)

    W0, W1 = update_weights(W0, W1, l0, l1, l2D, l1D)

#saves W matrixes in text files 
W0_save = np.savetxt('W0.txt', W0)
W1_save = np.savetxt('W1.txt', W1)

print ("Output after training")
print (l2)


# Plotting cost function v. iterations (uses Bokeh)
output_file("Error.html")
plot = figure(plot_width = 800, plot_height = 800)
plot.outline_line_width = 5
plot.outline_line_color = 'black'

plot.xaxis.axis_label = 'Number of iterations'
plot.xaxis.axis_label_text_color = 'black'
plot.xaxis.axis_label_text_font_size = '22pt'
plot.xaxis.axis_label_text_font_style = 'bold'

plot.yaxis.axis_label = 'Error'
plot.yaxis.axis_label_text_color = 'black'
plot.yaxis.axis_label_text_font_size = '22pt'
plot.yaxis.axis_label_text_font_style = 'bold'

plot.line(num_iter, error, line_width = 3, color = 'navy')

show(plot)


