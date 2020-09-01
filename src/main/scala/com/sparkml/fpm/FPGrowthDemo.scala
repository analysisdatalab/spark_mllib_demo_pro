package com.sparkml.fpm

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.fpm.{AssociationRules, FPGrowth, FPGrowthModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer

/**
 * 关联规则算法--看了又看,买了又买案例-使用FPGrowth生成关联规则,实现看了又看,买了又买功能
 */
object FPGrowthDemo {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession.builder().appName("FPGrowthDemo").master("local").getOrCreate()
    val sc = spark.sparkContext

    //todo 加载数据
    val rdd = sc.textFile("data/association/sample_fpgrowth.txt")
      .map(line => line.split("\\s+"))

    val fpg = new FPGrowth()
    //设置最小支持度
    fpg.setMinSupport(0.5)
    //设置分区数
    fpg.setNumPartitions(2)

    //todo 使用fpg算法进行训练
    /**
     * def run[Item: ClassTag](data: RDD[Array[Item]]): FPGrowthModel[Item]
     *
     * r z h k p==>Array(r,z,h,k,p)
     */
    val fpgModel: FPGrowthModel[String] = fpg.run(rdd)


    //todo 可以通过fpgModel拿到频繁项集
    /**
     * class FreqItemset[Item] @Since("1.3.0") (
     * val items: Array[Item], 频繁项 x,y
     * val freq: Long  频繁项出现的次数
     * ) extends Serializable
     */
    val itemsetsRDD: RDD[FPGrowth.FreqItemset[String]] = fpgModel.freqItemsets

    /**
     * {t}: 3
     * {t,x}: 3
     * {t,x,z}: 3
     * {t,z}: 3
     * {s}: 3
     * {s,x}: 3
     * {z}: 5
     * {y}: 3
     * {y,t}: 3
     * ........
     */
    itemsetsRDD.collect().foreach(println)
    println("=========================================")
    //todo 使用关联规则模型生成关联规则，可以设置最小置信度，过滤不满足条件的频繁项集
    /** Confidence（尿布==>啤酒）=800/1000=0.8
     * class Rule[Item] private[fpm] (
     * val antecedent: Array[Item],//前项 尿布
     * val consequent: Array[Item],//后项 啤酒
     * freqUnion: Double,//前项和后项共同出现的次数，相当于尿布和啤酒同时购买的数量 800
     * freqAntecedent: Double, //前项出现的次数，相当于尿布购买的数量1000
     * freqConsequent: Option[Double]//后项出现的次数，可以忽略
     * )
     * 计算置信度   置信度=前后项共同出现的次数 / 前项出现的次数
     * def confidence: Double = freqUnion / freqAntecedent
     */
    val rulesRDD: RDD[AssociationRules.Rule[String]] = fpgModel.generateAssociationRules(0.8)
    rulesRDD.foreach(println)
    println("============================================================================================")

    /**
     * (t,(y,1.0))
     * (y_t_x,(z,1.0))
     * (y_z,(t,1.0))
     * (y_z,(x,1.0))
     * (t_x,(z,1.0))
     * (t_x,(y,1.0))
     *
     * (y_z,(t,1.0))
     * (y_z,(x,1.0))
     *
     * (y_z,list((t,1.0),(x,1.0)))
     */
    val tuple2RDD = rulesRDD.map(rule => {
      //(前项，（后项，置信度）)
      (rule.antecedent.mkString("_"), (rule.consequent.mkString("_"), rule.confidence))
    })

    val aggregateRDD = tuple2RDD.aggregateByKey(ListBuffer[(String, Double)]())(
      //这个函数是对分区进行统计
      (u, v) => {
        //ListBuffer[(String, Double)]()
        u += v
        u.sortBy(_._2).take(5)
      },
      //这个函数是对各个分区的结果进行归并
      (u1, v1) => {
        u1 ++= v1
        u1.sortBy(_._2).take(5)
      })

    /**
     * (y_t,ListBuffer((x,1.0), (z,1.0)))
     * (s,ListBuffer((x,1.0)))
     * (y_x_z,ListBuffer((t,1.0)))
     * (y,ListBuffer((t,1.0), (x,1.0), (z,1.0)))
     */
    aggregateRDD.foreach(println)
    println("============================================================================================")
    aggregateRDD.filter(t2 => t2._1.split("_").length == 1).foreach(t2 => {
      println(s"浏览此商品${t2._1}的顾客也同时浏览")
      t2._2.foreach(x2 => {
        println(s"商品名：${x2._1} 置信度：${x2._2}")
      })
      println("-----------------------------------------------")
    })


    spark.stop()
  }
}
