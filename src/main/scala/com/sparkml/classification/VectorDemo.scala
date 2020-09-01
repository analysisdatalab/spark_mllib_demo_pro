package com.sparkml.classification

import org.apache.spark.mllib.linalg.Vectors

/**
 *
 */
object VectorDemo {
  def main(args: Array[String]): Unit = {
    //创建一个稀疏向量; 大小为3，脚标0的数据为4，脚标为2的数据为5，脚标为1的数据为0
    val vector = Vectors.sparse(3, Array(0, 2), Array(4, 5))
    println(vector)
    println(vector(0))
    println(vector(1))
    println(vector(2))
    //创建稠密向量
    val vector1 = Vectors.dense(1, 2, 3, 4, 5)
    println(vector1)
  }
}
