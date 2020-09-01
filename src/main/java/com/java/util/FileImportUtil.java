package com.java.util;

import org.apache.poi.POIXMLDocument;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.hwpf.HWPFDocument;
import org.apache.poi.openxml4j.opc.OPCPackage;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.apache.poi.xwpf.extractor.XWPFWordExtractor;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class FileImportUtil {

    private static final String DEFAULT_CHARSET = "GB2312";


    /**
     * 读取文件统一入口
     *
     * @param path
     * @return
     */
    public static String importFile(String path, String charset) {
        String resTextString = "";
        if (path == null) {
            return resTextString;
        }
        try {
            int lastIndexOf = path.lastIndexOf(".");
            String type = path.substring(lastIndexOf + 1);
            if ("txt".equals(type)) {
                resTextString = importTxt(path, charset);
            } else if ("doc".equals(type)) {
                resTextString = importWord3(path);
            } else if ("docx".equals(type)) {
                resTextString = importWord7(path);
            } else if ("xls".equals(type) || "xlsx".equals(type)) {
                resTextString = importExcel(path, charset);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return resTextString;
    }


    /**
     * 读取word2003
     *
     * @throws
     */
    public static String importWord3(String path) {
        String textString = null;
        FileInputStream inputStream = null;
        try {
            inputStream = new FileInputStream(path);
            HWPFDocument doc = new HWPFDocument(inputStream);
            textString = doc.getText().toString();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return textString;
    }

    /**
     * 读取word2007
     *
     * @param path
     * @return
     */
    public static String importWord7(String path) {
        String text = null;
        OPCPackage openPackage = null;
        try {
            openPackage = POIXMLDocument.openPackage(path);
            XWPFWordExtractor word = new XWPFWordExtractor(openPackage);
            text = word.getText();
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("无内容：" + path);
        } finally {
            if (openPackage != null) {
                try {
                    openPackage.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return text;
    }

    /**
     * 导入txt 文件
     *
     * @param path
     * @return
     */
    // try-with-resources可以优雅关闭文件，异常时自动关闭文件
    public static String importTxt(String path, String charset) {
        String resText = "";
        if (charset == null) {
            charset = DEFAULT_CHARSET;
        }
        InputStreamReader reader = null;
        try {
            reader = new InputStreamReader(new FileInputStream(path), charset);
            BufferedReader br = new BufferedReader(reader);
            String line;
            while ((line = br.readLine()) != null) {
                resText += line;
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
        return resText;
    }

    /**
     * 导入.xls文档
     *
     * @param path
     * @param charset
     * @return
     */
    public static String importExcel(String path, String charset) {
        String resString = "";
        Workbook wb = null;
        InputStream is = null;
        int lastIndexOf = path.lastIndexOf(".");
        String type = path.substring(lastIndexOf + 1);
        try {
            is = new FileInputStream(path);
            if ("xls".equals(type)) {
                wb = new HSSFWorkbook(is);
            } else if ("xlsx".equals(type)) {
                wb = new XSSFWorkbook(is);
            } else {
                return resString;
            }
            if (wb != null) {
                resString = getExcelVal(wb);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (is != null) {
                try {
                    is.close();
                } catch (Exception e2) {
                    e2.printStackTrace();
                }
            }
        }
        return resString;
    }

    /**
     * 获取值
     */
    public static String getExcelVal(Workbook wb) {
        Sheet sheet = null;
        Row row = null;
        String cellData = null;
        String resString = "";
        //用来存放表中数据
        List<ArrayList<String>> list = new ArrayList<ArrayList<String>>();
        //Excel的页签数量
        int sheetNum = wb.getNumberOfSheets();
        for (int m = 0; m < sheetNum; m++) {
            //获取一个sheet
            sheet = wb.getSheetAt(m);
            Sheet sheetAt = wb.getSheetAt(0);
            //获取最大行数
            int rownum = sheet.getPhysicalNumberOfRows();
            //获取第一行
            row = sheet.getRow(0);
            if (row == null) {
                continue;
            }
            //获取最大列数
            int colnum = row.getPhysicalNumberOfCells();
            for (int i = 0; i < rownum; i++) {
                row = sheet.getRow(i);
                ArrayList<String> colList = new ArrayList<String>();
                if (row != null) {
                    for (int j = 0; j < colnum; j++) {
                        cellData = String.valueOf(getCellFormatValue(row.getCell(j)));
                        colList.add(cellData);
                        if (j < colnum - 1) {
                            resString += cellData + " ";
                        } else {
                            resString += cellData + "\n";
                        }
                    }
                } else {
                    break;
                }
                list.add(colList);
            }
        }
        return resString;
    }

    /**
     * 根据类型取值
     *
     * @param cell
     * @return
     */
    @SuppressWarnings("deprecation")
    public static Object getCellFormatValue(Cell cell) {
        Object cellValue = null;
        if (cell != null) {
            //判断cell类型
            switch (cell.getCellType()) {
                case Cell.CELL_TYPE_NUMERIC: {
                    cellValue = String.valueOf(cell.getNumericCellValue());
                    break;
                }
                case Cell.CELL_TYPE_FORMULA: {
                    //判断cell是否为日期格式
                    if (DateUtil.isCellDateFormatted(cell)) {
                        //转换为日期格式YYYY-mm-dd
                        cellValue = cell.getDateCellValue();
                    } else {
                        //数字
                        cellValue = String.valueOf(cell.getNumericCellValue());
                    }
                    break;
                }
                case Cell.CELL_TYPE_STRING: {
                    cellValue = cell.getRichStringCellValue().getString();
                    break;
                }
                default:
                    cellValue = "";
            }
        } else {
            cellValue = "";
        }
        return cellValue;
    }
}
