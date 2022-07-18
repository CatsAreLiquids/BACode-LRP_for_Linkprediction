import importlib
import igraph
import numpy
import matplotlib.pyplot as plt
import numpy as np
import scipy
import random

def shrink(rx, ry):
    rx = numpy.array(rx)
    ry = numpy.array(ry)


    rx = 0.75 * rx + 0.25 * rx.mean()
    ry = 0.75 * ry + 0.25 * ry.mean()

    rx = numpy.concatenate([
        numpy.linspace(rx[0], rx[0], 41),
        numpy.linspace(rx[0], rx[1], 20),
        numpy.linspace(rx[1], rx[2], 20),
        numpy.linspace(rx[2], rx[2], 41), ])
    ry = numpy.concatenate([
        numpy.linspace(ry[0], ry[0], 41),
        numpy.linspace(ry[0], ry[1], 20),
        numpy.linspace(ry[1], ry[2], 20),
        numpy.linspace(ry[2], ry[2], 41)])


    filt = numpy.exp(-numpy.linspace(-2, 2, 41) ** 2)
    filt = filt / filt.sum()

    rx = numpy.convolve(rx, filt, mode='valid')
    ry = numpy.convolve(ry, filt, mode='valid')

    return rx, ry


def walks(A):
    w = []

    for v1 in numpy.arange(len(A)):
        for v2 in numpy.where(A[v1])[0]:
            for v3 in numpy.where(A[v2])[0]:
                w += [(v1, v2, v3)]

    return w


def layout(A):
    graph = igraph.Graph()
    graph.add_vertices(len(A))
    graph.add_edges(zip(*numpy.where(A == 1)))
    return numpy.array(list(graph.layout_kamada_kawai()))

