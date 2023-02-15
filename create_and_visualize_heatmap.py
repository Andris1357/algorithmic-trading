import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import time

scatter_gap, newl = 5, '\n'
scales = pd.read_csv("C:\\Users\\andri\\Documents\\Programming\\matic_scales.csv").iloc[:, 1:]
features = ['change', 'volume', 'price']
lookup_df = pd.read_excel("C:\\Users\\andri\\Documents\\Programming\\matic_py_lookup.xlsx",
                          sheet_name= "matic_py_lookup").iloc[:, 1:]

scales = scales.iloc[:275, :] # Change hardcoded index ranges to be able test on a reduced dataset

start: float = time.perf_counter()
rank: int = scales.shape[1]
lookup_length: int = lookup_df.shape[0]
scale_length: int = scales.shape[0]

T_lookup_vals: tf.TensorSpec(tf.TensorShape([None, 1])) = tf.constant(
    [[float(lookup_df.iat[row_i, lookup_df.shape[1] - 1])] for row_i in range(lookup_length)]
, dtype=tf.float64)

T_lookup_indices: tf.TensorSpec(tf.TensorShape([None, 1, 3])) = tf.reshape(tf.constant(
    [lookup_df.iloc[row_i, 1: lookup_df.shape[1]].astype(int) for row_i in range(lookup_length)]
    , dtype=tf.float64
), [lookup_length, 1, rank])
del lookup_df

T_heatmap = tf.reshape(tf.convert_to_tensor(
                np.arange(scale_length ** rank), dtype= tf.float64
            ), [scale_length for _ in range(rank)])

@tf.function(jit_compile= False)
def tfMapParallel(func_: tf.function, cube_: tf.Tensor) -> tf.Tensor:
    return tf.map_fn(func_, cube_, parallel_iterations= 20)

@tf.function
def generateCoordinates(coordinate_index_: tf.TensorSpec(tf.TensorShape([]))) -> tf.TensorSpec(tf.TensorShape([3,])):
    return tf.stack([
        tf.math.floordiv( # for 1st (depth) coordinate
            tf.cast(coordinate_index_, tf.float64),
            tf.pow(tf.constant(scale_length, dtype= tf.float64), tf.constant(2., dtype= tf.float64))
        ),
        tf.math.floordiv(
            tf.subtract(coordinate_index_, tf.multiply(
                tf.math.floordiv(coordinate_index_, tf.constant(scale_length ** 2, dtype= tf.float64)),
                tf.constant(scale_length ** 2, dtype=tf.float64)
            )),
            tf.constant(scale_length, dtype= tf.float64)
        ),
        tf.subtract(
            tf.cast(coordinate_index_, tf.float64),
            tf.multiply(
                tf.math.floordiv(tf.cast(coordinate_index_, tf.float64), tf.constant(scale_length, dtype= tf.float64)),
                tf.constant(scale_length, dtype=tf.float64)
            ) 
        )
    ], axis= 0)

def vectorCoordinates(vec_: tf.TensorSpec(tf.TensorShape([None, ]))) -> tf.TensorSpec(tf.TensorShape([None, 3])):
    return tfMapParallel(generateCoordinates, vec_)

def distanceSum(vec_: tf.TensorSpec(tf.TensorShape([None, ]))) -> tf.TensorSpec(tf.TensorShape([None, 1, None])):
    return tf.reduce_sum(tf.pow(
        tf.subtract(vectorCoordinates(vec_), T_lookup_indices),
        tf.constant(2, dtype= tf.float64)
    ), axis= 2)

def calculateWeights(vec_: tf.TensorSpec(tf.TensorShape([None, ]))) -> tf.TensorSpec(tf.TensorShape([None, ])):
    return tf.reduce_sum(tf.multiply(
        tf.divide(tf.constant(1., dtype=tf.float64), tf.pow(
            distanceSum(vec_),
            tf.constant(0.5, dtype=tf.float64)
        )),
        T_lookup_vals
    ), axis= 0)

def iterateMatrix(mx_: tf.TensorSpec(tf.TensorShape([None, None]))) -> tf.TensorSpec(tf.TensorShape([None, ])):
    return tfMapParallel(calculateWeights, mx_)


print(f"scale length => time elapsed => complexity: {scale_length ** rank}=>{time.perf_counter() - start}=>"
      f"{scale_length * rank * T_lookup_indices.shape[0] * 2}{newl}Heatmap weight: {tf.reduce_sum(T_heatmap)}")

x_src, y_src, z_src = [scales.iloc[:, x] for x in range(rank)]
vals = tf.reshape(T_heatmap[:, :, ::scatter_gap], [T_heatmap.shape[0] ** rank // scatter_gap]).numpy().tolist()
print("vals done")
xs_static = [[x_src[x % T_heatmap.shape[0]] for _ in range(T_heatmap.shape[0] // scatter_gap)] for x in range(T_heatmap.shape[0] ** 2)]
print("xs done")
zs_static = [[z_src[x // T_heatmap.shape[0]] for _ in range(T_heatmap.shape[0] // scatter_gap)] for x in range(T_heatmap.shape[0] ** 2)]
print("zs done")
ys_dynamic = [
    [y_src[x2] for x2 in range(x % scatter_gap, T_heatmap.shape[0], scatter_gap)][
        :T_heatmap.shape[0] // scatter_gap
    ] for x in range(T_heatmap.shape[0] ** 2)
]
print("ys done")

fig, axes = plt.figure(), plt.axes(projection='3d')
del T_heatmap
axes.scatter3D(xs_static, ys_dynamic, zs_static, c=vals, cmap='rainbow')
axes.set_xlabel(features[0], fontweight='bold')
axes.set_ylabel(features[1], fontweight='bold')
axes.set_zlabel(features[2], fontweight='bold')
plt.show()
