import math
import matplotlib
import matplotlib.pyplot as plt
import numpy

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def main():
    iterations = numpy.arange(500)

    rand = numpy.zeros(500)
    #for _ in range(12):
        #rand += numpy.random.rand(500)
    #rand = rand/12

    rand = numpy.random.rand(500)

    data = 1.0 - (0.5 * numpy.sin(2 * math.pi * iterations/500 * 4) + 0.5)
    start_data = 0.4 + (0.6 * data)
    end_data = 0.0 + (0.6 * data)

    data = start_data + (end_data - start_data) * rand

    iterations = iterations * 20

    #plt.plot(iterations, start_data)
    plt.plot(iterations, data, '0.6')
    #plt.plot(iterations, (end_data + start_data)/2.0, '0.0')
    plt.ylabel('Mixing Parameter (x)')
    plt.xlabel('Iterations')

    plt.savefig('rainfall_example.png', format='png')


if __name__ == '__main__':
    main()