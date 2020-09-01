package com.java.sparkml.ml.textclassification;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
import org.wltea.analyzer.app.WordCountAnalyse;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

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

public class T05_LoadModel {
    public static void main(String[] args) throws IOException {
        Logger.getLogger("org").setLevel(Level.WARN);
        // 1.初始化spark配置信息并建立与spark的连接
        SparkSession spark = SparkSession.builder().appName("T12_StockNewsClassification_full").master("local").getOrCreate();

        // Input data: Each row is a bag of words from a sentence or document.
        String path = args[0];
        InputStreamReader reader = null;
        List<Row> data = new ArrayList<>();
        try {
            reader = new InputStreamReader(new FileInputStream(path), "utf-8");
            BufferedReader br = new BufferedReader(reader);
            String line;
            while ((line = br.readLine()) != null) {
                String[] splits = line.split("\\[-\\]");
                Integer label = Integer.parseInt(splits[0]);

                Row row = RowFactory.create(label, WordCountAnalyse.splitWord(splits[1]));
                data.add(row);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
//        List<Row> data = Arrays.asList(
//                RowFactory.create(1,WordCountAnalyse.splitWord("年内9家上市公司因财务造假被处罚")),
//                RowFactory.create(1,WordCountAnalyse.splitWord("雷声滚滚！刚刚两家公司确定退市，事涉IPO造假、并购掉坑，18万股东中招")),
//                RowFactory.create(1, WordCountAnalyse.splitWord("安永派驻强大反舞弊团队后 瑞幸造假东窗事发")),
//                RowFactory.create(1, WordCountAnalyse.splitWord("自曝财务造假 瑞幸咖啡暴跌75.57%")),
//                RowFactory.create(1, WordCountAnalyse.splitWord("身陷财务造假漩涡 易见股份大跌后收问询函")),
//
//                RowFactory.create(1,WordCountAnalyse.splitWord("三峡人寿违规销售产品组合 被叫停新产品备案六个月")),
//                RowFactory.create(1,WordCountAnalyse.splitWord("又一股跌停！ST群兴因信披违规遭立案调查")),
//
//                RowFactory.create(1,WordCountAnalyse.splitWord("银河证券：海南自由贸易港迎长期投资机会，短期利好旅游业复苏")),
//                RowFactory.create(1,WordCountAnalyse.splitWord("小微金融政策出炉，中小银行迎结构性利好")),
//                RowFactory.create(1,WordCountAnalyse.splitWord("鹏博士大涨5.16% 首季净利润增长52.77%-60.52%")),
//                RowFactory.create(1,WordCountAnalyse.splitWord("智飞生物大涨5.30% 首季净利润增长0-10.00%")),
//                RowFactory.create(1,WordCountAnalyse.splitWord("27日强势板块分析：黄金概念爆发 医药板块再度活跃")),
//                RowFactory.create(1,WordCountAnalyse.splitWord("中国500强谁最能赚钱？工行第一，阿里腾讯进入前十"))
//        );
        StructType schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("message", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });

        Dataset<Row> testTitleDS = spark.createDataFrame(data, schema);
        Dataset<Row> filter = testTitleDS.filter(new FilterFunction<Row>() {
            @Override
            public boolean call(Row row) throws Exception {
                int label = row.getInt(0);
                if (label == 5) {
                    return false;
                } else {
                    return true;
                }
            }
        });
        // 需要进一步将切分单词的数据，转换为特征向量

        // TODO 1.加载管道与模型
        PipelineModel pipelineModel = PipelineModel.load("ml_model/pipeLineModel");

        // TODO 2.使用训练好的模型测试样例数据。使用持久化后的PipelineModel对原始数据进行预测
//        Dataset<Row> res = pipelineModel.transform(filter);
//        res.printSchema();
//        res.select("label", "predictedLabel").show(false);


        Dataset<Row> predictionResultDF = pipelineModel.transform(filter);

        //below 2 lines are for debug use
        predictionResultDF.printSchema();
        predictionResultDF.select( "label", "predictedLabel").show(false);



        //9， 评估训练结果
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double predictionAccuracy = evaluator.evaluate(predictionResultDF);
        System.out.println("Testing Accuracy is %2.4f".format(String.valueOf(predictionAccuracy * 100)) + "%");

    }
}


