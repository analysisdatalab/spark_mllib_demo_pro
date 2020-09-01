package com.java.sparkml.ml.textclassification;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;

/**
 * @Author:
 * @Date: 2020-07-24 15:04
 * @Version: 1.0
 * @Modified By:
 * @Description:
 */
/*
从 HDFS 上读取原始数据集，并创建一个 DataFrame。
使用 StringIndexer 将原始的文本标签 (“Ham”或者“Spam”) 转化成数值型的表型，以便 Spark ML 处理。
使用 Word2Vec 将短信文本转化成数值型词向量。
使用 MultilayerPerceptronClassifier 训练一个多层感知器模型。
使用 LabelConverter 将预测结果的数值标签转化成原始的文本标签。
最后在测试数据集上测试模型的预测精确度。

 */

// 参考链接：https://blog.csdn.net/weixin_32265569/article/details/107604269

public class T01_BadMailClassification {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.WARN);
        SparkSession spark = SparkSession.builder().appName("T08_BadMailClassification").master("local").getOrCreate();

        //2， 创建数据集并分词
        RDD<String> rdd = spark.sparkContext().textFile("data/ml/crossvalidation/垃圾邮件分类_SMSSpamCollection.txt", 2);
        JavaRDD<Row> parsedRDD = rdd.toJavaRDD().map(line -> {
            String[] splits = line.split(" ");
            return RowFactory.create(splits[0], splits[1].split(" "));
            /*if (splits[0].equals("ham")) {
                return RowFactory.create(0, splits[1].split(" "));
            } else {
                return RowFactory.create(1, splits[1].split(" "));
            }*/
        });


        // 第二步，动态构造元数据
        // 比如说，id、name等，field的名称和类型，可能都是在程序运行过程中，动态从mysql db里
        // 或者是配置文件中，加载出来的，是不固定的
        // 所以特别适合用这种编程的方式，来构造元数据
        StructType structType = new StructType(new StructField [] {
                new StructField("label", DataTypes.StringType, false, Metadata.empty()),
                new StructField("message", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });

        Dataset<Row> msgDF = spark.createDataFrame(parsedRDD, structType);

        //3， 将标签转化为索引值
        StringIndexerModel labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(msgDF);
        //4， 创建Word2Vec，分词向量大小100
        final int VECTOR_SIZE = 100;
        Word2Vec word2Vec = new Word2Vec().setInputCol("message").setOutputCol("features").setVectorSize(VECTOR_SIZE).setMinCount(1);

        //5， 创建多层感知器
        //输入层VECTOR_SIZE个，中间层两层分别是6，,5个神经元，输出层2个
        int[] layers = new int[] {VECTOR_SIZE, 6, 5, 2};
        MultilayerPerceptronClassifier mlpc = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(512)
                .setSeed(1234L).setMaxIter(128).setFeaturesCol("features").setLabelCol("indexedLabel").setPredictionCol("prediction");

        //6， 将索引转换为原有标签
        IndexToString labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels());

        //7， 数据集分割
        Dataset<Row>[] needFitData = msgDF.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainingData = needFitData[0];
        Dataset<Row> testData = needFitData[1];

        //8， 创建pipeline并训练数据
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{labelIndexer, word2Vec, mlpc, labelConverter});
        PipelineModel model = pipeline.fit(trainingData);
        Dataset<Row> predictionResultDF = model.transform(testData);

        //below 2 lines are for debug use
        predictionResultDF.printSchema();
        predictionResultDF.select("message", "label", "predictedLabel").show(30);

        //9， 评估训练结果
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double predictionAccuracy = evaluator.evaluate(predictionResultDF);
        System.out.println("Testing Accuracy is %2.4f".format(String.valueOf(predictionAccuracy * 100)) + "%");

        //将测试集数据交给管道进行测试
//    val predictions = model.transform(testData)
//    predictions.select("message","label", "predictedLabel").show(false)
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("label")
//      .setPredictionCol("predictedLabel")
//      .setMetricName("accuracy")
//    val accuracy = evaluator.evaluate(predictions)
//    println(s"正确率 = ${accuracy}" + s"\t错误率 = ${(1.0 - accuracy)}")
    }
}


