from PyQt5 import QtCore, QtGui, QtWidgets
import PreProcess
import AdjacencyMatrix
import ScoreAA
import ScoreCN
import ScoreJC
import ScorePA
import MachineLearning
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import svm

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setEnabled(True)
        Form.resize(741, 628)
        font = QtGui.QFont()
        font.setFamily("Batang")
        font.setBold(False)
        font.setWeight(50)
        Form.setFont(font)
        Form.setWindowTitle("Link Prediction with Machine Learning Approach")
        Form.setStyleSheet("QWidget{\n"
"    \n"
"    background-color: #ffffff;\n"
"}")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(60, 40, 311, 21))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setStyleSheet("#label{\n"
"    color: #263238;\n"
"    font: 20pt;\n"
"}\n"
"QLabel{\n"
"    font:\"Helvetica Neue\";\n"
"}\n"
"")
        self.label.setObjectName("label")
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setGeometry(QtCore.QRect(50, 110, 651, 451))
        self.tabWidget.setStyleSheet("QTabWidget::pane {\n"
"    background: white;\n"
"border-style: outset;\n"
"}\n"
"\n"
"QTabWidget::tab-bar:top {\n"
"    top: 1px;\n"
"}\n"
"\n"
"QTabWidget::tab-bar:bottom {\n"
"    bottom: 1px;\n"
"}\n"
"\n"
"QTabWidget::tab-bar:left {\n"
"    right: 1px;\n"
"}\n"
"\n"
"QTabWidget::tab-bar:right {\n"
"    left: 1px;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"    border-style: outset;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"    color: #263238;\n"
"}\n"
"\n"
"QTabBar::tab:!selected {\n"
"    color: #78909c;\n"
"}\n"
"\n"
"QTabBar::tab:!selected:hover {\n"
"    color: #03a9f4;\n"
"}\n"
"\n"
"QTabBar::tab:top:!selected {\n"
"    margin-top: 3px;\n"
"}\n"
"\n"
"QTabBar::tab:bottom:!selected {\n"
"    margin-bottom: 3px;\n"
"}\n"
"\n"
"QTabBar::tab:top, QTabBar::tab:bottom {\n"
"    min-width: 8ex;\n"
"    margin-right: -1px;\n"
"    padding: 5px 10px 5px 10px;\n"
"}\n"
"\n"
"QTabBar::tab:top:selected {\n"
"    border-style: outset;\n"
"}\n"
"\n"
"QTabBar::tab:bottom:selected {\n"
"    border-style: outset;\n"
"}\n"
"\n"
"QTabBar::tab:top:last, QTabBar::tab:bottom:last,\n"
"QTabBar::tab:top:only-one, QTabBar::tab:bottom:only-one {\n"
"    margin-right: 0;\n"
"}\n"
"\n"
"QTabBar::tab:left:!selected {\n"
"    margin-right: 3px;\n"
"}\n"
"\n"
"QTabBar::tab:right:!selected {\n"
"    margin-left: 3px;\n"
"}\n"
"\n"
"QTabBar::tab:left, QTabBar::tab:right {\n"
"    min-height: 8ex;\n"
"    margin-bottom: -1px;\n"
"    padding: 10px 5px 10px 5px;\n"
"}\n"
"\n"
"QTabBar::tab:left:selected {\n"
"   border-style: outset;\n"
"}\n"
"\n"
"QTabBar::tab:right:selected {\n"
"    border-style: outset;\n"
"}\n"
"\n"
"QTabBar::tab:left:last, QTabBar::tab:right:last,\n"
"QTabBar::tab:left:only-one, QTabBar::tab:right:only-one {\n"
"    margin-bottom: 0;\n"
"}")
        self.tabWidget.setObjectName("tabWidget")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.line_2 = QtWidgets.QFrame(self.tab_4)
        self.line_2.setGeometry(QtCore.QRect(10, 0, 118, 3))
        self.line_2.setStyleSheet("background-color: #03a9f4;")
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_6 = QtWidgets.QLabel(self.tab_4)
        self.label_6.setGeometry(QtCore.QRect(30, 50, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("#label_6{\n"
"    font: 14pt;\n"
"    color: #263238;\n"
"    \n"
"}")
        self.label_6.setObjectName("label_6")
        self.label_4 = QtWidgets.QLabel(self.tab_4)
        self.label_4.setGeometry(QtCore.QRect(30, 180, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("#label_4{\n"
"    color: #78909c;\n"
"    font: 14pt;\n"
"}\n"
"")
        self.label_4.setObjectName("label_4")
        self.comboBox = QtWidgets.QComboBox(self.tab_4)
        self.comboBox.setGeometry(QtCore.QRect(30, 210, 221, 31))
        self.comboBox.setStyleSheet("QComboBox\n"
"{\n"
"    selection-background-color: #03a9f4;\n"
"    border: 2px solid #eceff1;\n"
"    border-radius: 1px;\n"
"}\n"
"\n"
"QComboBox:hover,QPushButton:hover\n"
"{\n"
"    border: 2px solid #03a9f4;\n"
"}\n"
"\n"
"\n"
"QComboBox:on\n"
"{\n"
"    padding-top: 3px;\n"
"    padding-left: 4px;\n"
"    selection-background-color: #03a9f4;\n"
"}\n"
"\n"
"QComboBox QAbstractItemView\n"
"{\n"
"    border: 2px solid darkgray;\n"
"    selection-background-color: #03a9f4;\n"
"}\n"
"\n"
"QComboBox::drop-down\n"
"{\n"
"     subcontrol-origin: padding;\n"
"     subcontrol-position: top right;\n"
"     width: 15px;\n"
"\n"
"     border-left-width: 0px;\n"
"     border-left-color: darkgray;\n"
"     border-left-style: solid; /* just a single line */\n"
"     border-top-right-radius: 3px; /* same radius as the QComboBox */\n"
"     border-bottom-right-radius: 3px;\n"
" }\n"
"\n"
"QComboBox::down-arrow\n"
"{\n"
"     image: url(:/down_arrow.png);\n"
"}")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.pushButton = QtWidgets.QPushButton(self.tab_4)
        self.pushButton.setGeometry(QtCore.QRect(50, 280, 186, 41))
        self.pushButton.setStyleSheet("QPushButton {\n"
"    background-color: #03a9f4;\n"
"    border-style: outset;\n"
"    border-color: #f3f5f7;\n"
"    border-radius: 20px;\n"
"    font: bold 14px;\n"
"    min-width: 10em;\n"
"    padding: 6px;\n"
"    color: #ffffff;\n"
"    font: 13pt \"Helvetica Neue\";\n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: #2980b9;\n"
"    border-style: outset;\n"
"    border-color: #f3f5f7;\n"
"\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.label_5 = QtWidgets.QLabel(self.tab_4)
        self.label_5.setGeometry(QtCore.QRect(30, 90, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("#label_5{\n"
"    color: #78909c;\n"
"    font: 14pt;\n"
"    \n"
"}\n"
"")
        self.label_5.setObjectName("label_5")
        self.graphicsView_4 = QtWidgets.QGraphicsView(self.tab_4)
        self.graphicsView_4.setGeometry(QtCore.QRect(10, 40, 261, 351))
        self.graphicsView_4.setStyleSheet("#graphicsView_4{\n"
"    border: 2px solid #eceff1;\n"
"    border-radius: 1px;\n"
"}")
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.graphicsView_5 = QtWidgets.QGraphicsView(self.tab_4)
        self.graphicsView_5.setGeometry(QtCore.QRect(290, 40, 341, 351))
        self.graphicsView_5.setStyleSheet("#graphicsView_5{\n"
"    border: 2px solid red;\n"
"    border-color: #eceff1;\n"
"    border-radius: 1px;\n"
"}")
        self.graphicsView_5.setObjectName("graphicsView_5")
        self.label_15 = QtWidgets.QLabel(self.tab_4)
        self.label_15.setGeometry(QtCore.QRect(335, 50, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_15.setFont(font)
        self.label_15.setStyleSheet("#label_6{\n"
"    font: 14pt;\n"
"    color: #263238;\n"
"    \n"
"}")
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.tab_4)
        self.label_16.setGeometry(QtCore.QRect(335, 70, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_16.setFont(font)
        self.label_16.setStyleSheet("#label_16{\n"
"    color: #78909c;\n"
"    font: 14pt;\n"
"    \n"
"}\n"
"")
        self.label_16.setTextFormat(QtCore.Qt.RichText)
        self.label_16.setWordWrap(True)
        self.label_16.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.tab_4)
        self.label_17.setGeometry(QtCore.QRect(335, 130, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_17.setFont(font)
        self.label_17.setStyleSheet("#label_6{\n"
"    font: 14pt;\n"
"    color: #263238;\n"
"    \n"
"}")
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.tab_4)
        self.label_18.setGeometry(QtCore.QRect(335, 150, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_18.setFont(font)
        self.label_18.setStyleSheet("#label_18{\n"
"    color: #78909c;\n"
"    font: 14pt;\n"
"}\n"
"")
        self.label_18.setTextFormat(QtCore.Qt.RichText)
        self.label_18.setWordWrap(True)
        self.label_18.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        self.label_18.setObjectName("label_18")
        self.label_34 = QtWidgets.QLabel(self.tab_4)
        self.label_34.setGeometry(QtCore.QRect(335, 210, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_34.setFont(font)
        self.label_34.setStyleSheet("#label_6{\n"
"    font: 14pt;\n"
"    color: #263238;\n"
"    \n"
"}")
        self.label_34.setObjectName("label_34")
        self.label_35 = QtWidgets.QLabel(self.tab_4)
        self.label_35.setGeometry(QtCore.QRect(335, 240, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_35.setFont(font)
        self.label_35.setStyleSheet("#label_35{\n"
"    color: #78909c;\n"
"    font: 14pt;\n"
"}\n"
"")
        self.label_35.setTextFormat(QtCore.Qt.RichText)
        self.label_35.setWordWrap(True)
        self.label_35.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        self.label_35.setObjectName("label_35")
        self.label_36 = QtWidgets.QLabel(self.tab_4)
        self.label_36.setGeometry(QtCore.QRect(335, 310, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_36.setFont(font)
        self.label_36.setStyleSheet("#label_6{\n"
"    font: 14pt;\n"
"    color: #263238;\n"
"    \n"
"}")
        self.label_36.setObjectName("label_36")
        self.label_37 = QtWidgets.QLabel(self.tab_4)
        self.label_37.setGeometry(QtCore.QRect(335, 330, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_37.setFont(font)
        self.label_37.setStyleSheet("#label_37{\n"
"    color: #78909c;\n"
"    font: 14pt;\n"
"}\n"
"")
        self.label_37.setTextFormat(QtCore.Qt.RichText)
        self.label_37.setWordWrap(True)
        self.label_37.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        self.label_37.setObjectName("label_37")
        self.progressBar_1 = QtWidgets.QProgressBar(self.tab_4)
        self.progressBar_1.setGeometry(QtCore.QRect(30, 350, 221, 23))
        self.progressBar_1.setProperty("value", 24)
        self.progressBar_1.setObjectName("progressBar_1")
        self.graphicsView_4.raise_()
        self.line_2.raise_()
        self.label_6.raise_()
        self.label_4.raise_()
        self.comboBox.raise_()
        self.pushButton.raise_()
        self.label_5.raise_()
        self.graphicsView_5.raise_()
        self.label_15.raise_()
        self.label_16.raise_()
        self.label_17.raise_()
        self.label_18.raise_()
        self.label_34.raise_()
        self.label_35.raise_()
        self.label_36.raise_()
        self.label_37.raise_()
        self.progressBar_1.raise_()
        self.tabWidget.addTab(self.tab_4, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.line_3 = QtWidgets.QFrame(self.tab)
        self.line_3.setGeometry(QtCore.QRect(145, 0, 110, 3))
        self.line_3.setStyleSheet("background-color: #03a9f4;")
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.label_38 = QtWidgets.QLabel(self.tab)
        self.label_38.setGeometry(QtCore.QRect(40, 60, 211, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_38.setFont(font)
        self.label_38.setStyleSheet("#label_6{\n"
"    font: 14pt;\n"
"    color: #263238;\n"
"    \n"
"}")
        self.label_38.setObjectName("label_38")
        self.radioButton = QtWidgets.QRadioButton(self.tab)
        self.radioButton.setGeometry(QtCore.QRect(70, 110, 141, 20))
        self.radioButton.setStyleSheet("QRadioButton{\n"
"    font:\"Helvetica Neue\";\n"
"    \n"
"}")
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.tab)
        self.radioButton_2.setGeometry(QtCore.QRect(390, 110, 151, 20))
        self.radioButton_2.setStyleSheet("QRadioButton{\n"
"    font:\"Helvetica Neue\";\n"
"    \n"
"}")
        self.radioButton_2.setObjectName("radioButton_2")
        self.graphicsView_11 = QtWidgets.QGraphicsView(self.tab)
        self.graphicsView_11.setGeometry(QtCore.QRect(10, 40, 621, 361))
        self.graphicsView_11.setStyleSheet("#graphicsView_11{\n"
"    border: 2px solid red;\n"
"    border-color: #eceff1;\n"
"    border-radius: 1px;\n"
"}")
        self.graphicsView_11.setObjectName("graphicsView_11")
        self.radioButton_3 = QtWidgets.QRadioButton(self.tab)
        self.radioButton_3.setEnabled(False)
        self.radioButton_3.setGeometry(QtCore.QRect(70, 200, 141, 20))
        self.radioButton_3.setStyleSheet("QRadioButton{\n"
"    font:\"Helvetica Neue\";\n"
"    \n"
"}")
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QtWidgets.QRadioButton(self.tab)
        self.radioButton_4.setEnabled(False)
        self.radioButton_4.setGeometry(QtCore.QRect(390, 200, 141, 20))
        self.radioButton_4.setStyleSheet("QRadioButton{\n"
"    font:\"Helvetica Neue\";\n"
"    \n"
"}")
        self.radioButton_4.setObjectName("radioButton_4")
        self.pushButton_3 = QtWidgets.QPushButton(self.tab)
        self.pushButton_3.setGeometry(QtCore.QRect(220, 300, 186, 41))
        self.pushButton_3.setStyleSheet("QPushButton {\n"
"    background-color: #03a9f4;\n"
"    border-style: outset;\n"
"    border-color: #f3f5f7;\n"
"    border-radius: 20px;\n"
"    font: bold 14px;\n"
"    min-width: 10em;\n"
"    padding: 6px;\n"
"    color: #ffffff;\n"
"    font: 13pt \"Helvetica Neue\";\n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: #2980b9;\n"
"    border-style: outset;\n"
"    border-color: #f3f5f7;\n"
"\n"
"}")
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_39 = QtWidgets.QLabel(self.tab)
        self.label_39.setGeometry(QtCore.QRect(100, 135, 201, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_39.setFont(font)
        self.label_39.setStyleSheet("#label_39{\n"
"    color: #78909c;\n"
"    font: 14pt;\n"
"    \n"
"}\n"
"")
        self.label_39.setWordWrap(True)
        self.label_39.setObjectName("label_39")
        self.label_40 = QtWidgets.QLabel(self.tab)
        self.label_40.setGeometry(QtCore.QRect(390, 135, 201, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_40.setFont(font)
        self.label_40.setStyleSheet("#label_40{\n"
"    color: #78909c;\n"
"    font: 14pt;\n"
"    \n"
"}\n"
"")
        self.label_40.setWordWrap(True)
        self.label_40.setObjectName("label_40")
        self.label_41 = QtWidgets.QLabel(self.tab)
        self.label_41.setGeometry(QtCore.QRect(100, 230, 201, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_41.setFont(font)
        self.label_41.setStyleSheet("#label_41{\n"
"    color: #78909c;\n"
"    font: 14pt;\n"
"    \n"
"}\n"
"")
        self.label_41.setWordWrap(True)
        self.label_41.setObjectName("label_41")
        self.label_65 = QtWidgets.QLabel(self.tab)
        self.label_65.setGeometry(QtCore.QRect(390, 230, 201, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_65.setFont(font)
        self.label_65.setStyleSheet("#label_65{\n"
"    color: #78909c;\n"
"    font: 14pt;\n"
"    \n"
"}\n"
"")
        self.label_65.setWordWrap(True)
        self.label_65.setObjectName("label_65")
        self.progressBar_2 = QtWidgets.QProgressBar(self.tab)
        self.progressBar_2.setGeometry(QtCore.QRect(50, 360, 541, 23))
        self.progressBar_2.setProperty("value", 24)
        self.progressBar_2.setObjectName("progressBar_2")
        self.graphicsView_11.raise_()
        self.line_3.raise_()
        self.label_38.raise_()
        self.radioButton.raise_()
        self.radioButton_2.raise_()
        self.radioButton_3.raise_()
        self.radioButton_4.raise_()
        self.pushButton_3.raise_()
        self.label_39.raise_()
        self.label_40.raise_()
        self.label_41.raise_()
        self.label_65.raise_()
        self.progressBar_2.raise_()
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.line_4 = QtWidgets.QFrame(self.tab_2)
        self.line_4.setGeometry(QtCore.QRect(270, 0, 148, 3))
        self.line_4.setStyleSheet("background-color: #03a9f4;")
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.graphicsView_18 = QtWidgets.QGraphicsView(self.tab_2)
        self.graphicsView_18.setGeometry(QtCore.QRect(10, 40, 621, 361))
        self.graphicsView_18.setStyleSheet("#graphicsView_18{\n"
"    border: 2px solid red;\n"
"    border-color: #eceff1;\n"
"    border-radius: 1px;\n"
"}")
        self.graphicsView_18.setObjectName("graphicsView_18")
        self.label_66 = QtWidgets.QLabel(self.tab_2)
        self.label_66.setGeometry(QtCore.QRect(40, 60, 241, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_66.setFont(font)
        self.label_66.setStyleSheet("#label_66{\n"
"    font: 14pt;\n"
"    color: #263238;\n"
"    \n"
"}")
        self.label_66.setObjectName("label_66")
        self.pushButton_6 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_6.setGeometry(QtCore.QRect(410, 190, 186, 41))
        self.pushButton_6.setStyleSheet("QPushButton {\n"
"    background-color: #03a9f4;\n"
"    border-style: outset;\n"
"    border-color: #f3f5f7;\n"
"    border-radius: 20px;\n"
"    font: bold 14px;\n"
"    min-width: 10em;\n"
"    padding: 6px;\n"
"    color: #ffffff;\n"
"    font: 13pt \"Helvetica Neue\";\n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: #2980b9;\n"
"    border-style: outset;\n"
"    border-color: #f3f5f7;\n"
"\n"
"}")
        self.pushButton_6.setObjectName("pushButton_6")
        self.checkBox = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox.setGeometry(QtCore.QRect(80, 100, 171, 20))
        self.checkBox.setStyleSheet("QCheckBox{\n"
"    font:\"Helvetica Neue\";\n"
"    \n"
"}")
        self.checkBox.setObjectName("checkBox")
        self.checkBox_2 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_2.setGeometry(QtCore.QRect(80, 140, 171, 20))
        self.checkBox_2.setStyleSheet("QCheckBox{\n"
"    font:\"Helvetica Neue\";\n"
"    \n"
"}")
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_3 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_3.setGeometry(QtCore.QRect(80, 180, 171, 20))
        self.checkBox_3.setStyleSheet("QCheckBox{\n"
"    font:\"Helvetica Neue\";\n"
"    \n"
"}")
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_4 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_4.setGeometry(QtCore.QRect(80, 220, 171, 20))
        self.checkBox_4.setStyleSheet("QCheckBox{\n"
"    font:\"Helvetica Neue\";\n"
"    \n"
"}")
        self.checkBox_4.setObjectName("checkBox_4")
        self.checkBox_5 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_5.setEnabled(False)
        self.checkBox_5.setGeometry(QtCore.QRect(80, 260, 261, 20))
        self.checkBox_5.setStyleSheet("QCheckBox{\n"
"    font:\"Helvetica Neue\";\n"
"    \n"
"}")
        self.checkBox_5.setObjectName("checkBox_5")
        self.checkBox_6 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_6.setEnabled(False)
        self.checkBox_6.setGeometry(QtCore.QRect(80, 300, 261, 20))
        self.checkBox_6.setStyleSheet("QCheckBox{\n"
"    font:\"Helvetica Neue\";\n"
"    \n"
"}")
        self.checkBox_6.setObjectName("checkBox_6")
        self.checkBox_7 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_7.setEnabled(False)
        self.checkBox_7.setGeometry(QtCore.QRect(80, 340, 261, 20))
        self.checkBox_7.setStyleSheet("QCheckBox{\n"
"    font:\"Helvetica Neue\";\n"
"    \n"
"}")
        self.checkBox_7.setObjectName("checkBox_7")
        self.progressBar_3 = QtWidgets.QProgressBar(self.tab_2)
        self.progressBar_3.setGeometry(QtCore.QRect(80, 370, 511, 23))
        self.progressBar_3.setProperty("value", 24)
        self.progressBar_3.setObjectName("progressBar_3")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.line_5 = QtWidgets.QFrame(self.tab_3)
        self.line_5.setGeometry(QtCore.QRect(436, 0, 71, 3))
        self.line_5.setStyleSheet("background-color: #03a9f4;")
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.label_67 = QtWidgets.QLabel(self.tab_3)
        self.label_67.setGeometry(QtCore.QRect(40, 60, 271, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_67.setFont(font)
        self.label_67.setStyleSheet("#label_66{\n"
"    font: 14pt;\n"
"    color: #263238;\n"
"    \n"
"}")
        self.label_67.setObjectName("label_67")
        self.graphicsView_19 = QtWidgets.QGraphicsView(self.tab_3)
        self.graphicsView_19.setGeometry(QtCore.QRect(10, 40, 621, 361))
        self.graphicsView_19.setStyleSheet("#graphicsView_19{\n"
"    border: 2px solid red;\n"
"    border-color: #eceff1;\n"
"    border-radius: 1px;\n"
"}")
        self.graphicsView_19.setObjectName("graphicsView_19")
        self.pushButton_7 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_7.setGeometry(QtCore.QRect(30, 150, 186, 41))
        self.pushButton_7.setStyleSheet("QPushButton {\n"
"    background-color: #03a9f4;\n"
"    border-style: outset;\n"
"    border-color: #f3f5f7;\n"
"    border-radius: 20px;\n"
"    font: bold 14px;\n"
"    min-width: 10em;\n"
"    padding: 6px;\n"
"    color: #ffffff;\n"
"    font: 13pt \"Helvetica Neue\";\n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: #2980b9;\n"
"    border-style: outset;\n"
"    border-color: #f3f5f7;\n"
"\n"
"}")
        self.pushButton_7.setObjectName("pushButton_7")
        self.radioButton_9 = QtWidgets.QRadioButton(self.tab_3)
        self.radioButton_9.setGeometry(QtCore.QRect(40, 100, 141, 20))
        self.radioButton_9.setStyleSheet("QRadioButton{\n"
"    font:\"Helvetica Neue\";\n"
"    \n"
"}")
        self.radioButton_9.setObjectName("radioButton_9")
        self.progressBar_4 = QtWidgets.QProgressBar(self.tab_3)
        self.progressBar_4.setGeometry(QtCore.QRect(30, 200, 581, 23))
        self.progressBar_4.setProperty("value", 24)
        self.progressBar_4.setObjectName("progressBar_4")
        self.plainTextEdit = QtWidgets.QTextEdit(self.tab_3)
        self.plainTextEdit.setGeometry(QtCore.QRect(30, 270, 581, 111))
        self.plainTextEdit.setStyleSheet("#plainTextEdit{\n"
"    border: 2px solid red;\n"
"    border-color: #eceff1;\n"
"    border-radius: 1px;\n"
"}")
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.label_68 = QtWidgets.QLabel(self.tab_3)
        self.label_68.setGeometry(QtCore.QRect(30, 240, 271, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_68.setFont(font)
        self.label_68.setStyleSheet("#label_66{\n"
"    font: 14pt;\n"
"    color: #263238;\n"
"    \n"
"}")
        self.label_68.setObjectName("label_68")
        self.radioButton_10 = QtWidgets.QRadioButton(self.tab_3)
        self.radioButton_10.setGeometry(QtCore.QRect(240, 100, 141, 20))
        self.radioButton_10.setStyleSheet("QRadioButton{\n"
"    font:\"Helvetica Neue\";\n"
"    \n"
"}")
        self.radioButton_10.setObjectName("radioButton_10")
        self.radioButton_11 = QtWidgets.QRadioButton(self.tab_3)
        self.radioButton_11.setEnabled(False)
        self.radioButton_11.setGeometry(QtCore.QRect(460, 100, 141, 20))
        self.radioButton_11.setStyleSheet("QRadioButton{\n"
"    font:\"Helvetica Neue\";\n"
"    \n"
"}")
        self.radioButton_11.setObjectName("radioButton_11")
        self.graphicsView_19.raise_()
        self.line_5.raise_()
        self.label_67.raise_()
        self.pushButton_7.raise_()
        self.radioButton_9.raise_()
        self.progressBar_4.raise_()
        self.plainTextEdit.raise_()
        self.label_68.raise_()
        self.radioButton_10.raise_()
        self.radioButton_11.raise_()
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.line_6 = QtWidgets.QFrame(self.tab_5)
        self.line_6.setGeometry(QtCore.QRect(524, 0, 94, 3))
        self.line_6.setStyleSheet("background-color: #03a9f4;")
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.label_7 = QtWidgets.QLabel(self.tab_5)
        self.label_7.setGeometry(QtCore.QRect(290, 40, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("#label_6{\n"
"    font: 14pt;\n"
"    color: #78909c;\n"
"    \n"
"    background-color: rgb(255, 255, 255);\n"
"color: #263238;\n"
"}")
        self.label_7.setObjectName("label_7")
        self.graphicsView = QtWidgets.QGraphicsView(self.tab_5)
        self.graphicsView.setGeometry(QtCore.QRect(10, 130, 201, 191))
        self.graphicsView.setStyleSheet("#graphicsView{\n"
"border: 2px solid red;\n"
"    border-color: #eceff1;\n"
"    border-radius: 1px;   \n"
"}")
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.tab_5)
        self.graphicsView_2.setGeometry(QtCore.QRect(230, 130, 201, 191))
        self.graphicsView_2.setStyleSheet("#graphicsView_2{\n"
"    border: 2px solid red;\n"
"    border-color: #eceff1;\n"
"    border-radius: 1px;\n"
"}")
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.tab_5)
        self.graphicsView_3.setGeometry(QtCore.QRect(450, 130, 201, 191))
        self.graphicsView_3.setStyleSheet("#graphicsView_3{\n"
"    border: 2px solid red;\n"
"    border-color: #eceff1;\n"
"    border-radius: 1px;\n"
"}")
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.label_8 = QtWidgets.QLabel(self.tab_5)
        self.label_8.setGeometry(QtCore.QRect(50, 240, 141, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("#label_8{\n"
"    font: 14pt;\n"
"    color: #263238;\n"
"    background-color: rgb(255, 255, 255);\n"
"}")
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.tab_5)
        self.label_9.setGeometry(QtCore.QRect(170, 70, 331, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("#label_9{\n"
"    font: 14pt;\n"
"    color: #78909c;\n"
"    \n"
"}")
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.tab_5)
        self.label_10.setGeometry(QtCore.QRect(80, 270, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_10.setFont(font)
        self.label_10.setStyleSheet("#label_10{\n"
"    background-color: rgb(255, 255, 255);\n"
"    font: 14pt;\n"
"    color: #78909c;\n"
"    \n"
"}")
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.tab_5)
        self.label_11.setGeometry(QtCore.QRect(240, 240, 181, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_11.setFont(font)
        self.label_11.setStyleSheet("#label_11{\n"
"    font: 14pt;\n"
"    color: #263238;\n"
"    background-color: rgb(255, 255, 255);\n"
"}")
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.tab_5)
        self.label_12.setGeometry(QtCore.QRect(300, 270, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_12.setFont(font)
        self.label_12.setStyleSheet("#label_12{\n"
"    background-color: rgb(255, 255, 255);\n"
"    font: 14pt;\n"
"    color: #78909c;\n"
"    \n"
"}")
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.tab_5)
        self.label_13.setGeometry(QtCore.QRect(524, 240, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_13.setFont(font)
        self.label_13.setStyleSheet("#label_13{\n"
"    font: 14pt;\n"
"    color: #263238;\n"
"    background-color: rgb(255, 255, 255);\n"
"}")
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.tab_5)
        self.label_14.setGeometry(QtCore.QRect(520, 270, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_14.setFont(font)
        self.label_14.setStyleSheet("#label_14{\n"
"    background-color: rgb(255, 255, 255);\n"
"    font: 14pt;\n"
"    color: #78909c;\n"
"    \n"
"}")
        self.label_14.setObjectName("label_14")
        self.tabWidget.addTab(self.tab_5, "")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(60, 70, 271, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("#label_3{\n"
"    color: #78909c;\n"
"    font: 14pt;\n"
"}\n"
"")
        self.label_3.setObjectName("label_3")
        self.line = QtWidgets.QFrame(Form)
        self.line.setGeometry(QtCore.QRect(0, 150, 781, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        #Add
        
        self.status_tab1 = False
        self.status_tab2 = False
        self.status_tab3 = False
        self.status_tab4 = False
        self.Path_File = None
        self.Matrix_Adjacency = []
        self.Feature_Vector = []
        self.pushButton.clicked.connect(self.tab1_Save)
        self.pushButton_3.clicked.connect(self.tab2_Save)
        self.pushButton_6.clicked.connect(self.tab3_Save)
        self.pushButton_7.clicked.connect(self.tab4_Save)
        QtCore.QMetaObject.connectSlotsByName(Form)

    #Add
    def tab1_Save(self):
        value_comboBox = self.comboBox.currentIndex()
        if value_comboBox == 0:
             self.progressBar_1.setValue(0)
             self.status_tab1 = False
             self.Path_File = None
        else:
             self.progressBar_1.setValue(100)
             self.status_tab1 = True
             text = str(self.comboBox.currentText())
             self.Path_File = 'Dataset/' + text + '.csv'

    def tab2_Save(self):
        if not self.radioButton.isChecked() and not self.radioButton_2.isChecked():
           return None
        Data = PreProcess.ReadDataNoWeight(self.Path_File)
        if self.radioButton.isChecked():
           self.Matrix_Adjacency = AdjacencyMatrix.Matrix_Link(Data)
        if self.radioButton_2.isChecked():
           self.Matrix_Adjacency = AdjacencyMatrix.Matrix_Link_Undirect(Data) 
        self.progressBar_2.setValue(100)
        print(self.Matrix_Adjacency, len(self.Matrix_Adjacency))       

    def tab3_Save(self):
        if self.checkBox.isChecked():
           self.Feature_Vector = ScoreCN.getMatrixHalf(self.Matrix_Adjacency)
        if self.checkBox_2.isChecked():
           self.Feature_Vector = ScoreJC.getMatrixHalf(self.Matrix_Adjacency)
        if self.checkBox_3.isChecked():
           self.Feature_Vector = ScoreAA.getMatrixHalf(self.Matrix_Adjacency)
        if self.checkBox_4.isChecked():
           self.Feature_Vector = ScorePA.getMatrixHalf(self.Matrix_Adjacency)
        self.progressBar_3.setValue(100)

    def tab4_Save(self):
        if self.radioButton_10.isChecked():
           X_train, X_test, y_train, y_test = train_test_split(self.Feature_Vector[:,:-1].reshape(-1,1), self.Feature_Vector[:,-1], test_size=0.2)
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1]}]

        #scores = ['precision', 'recall']
        scores = ['precision']
        for score in scores:
                clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
                clf.fit(X_train, y_train)
                means = clf.cv_results_['mean_test_score']
                stds = clf.cv_results_['std_test_score']
                print("Detailed classification report:")
                y_true, y_pred = y_test, clf.predict(X_test)
                report = str(classification_report(y_true, y_pred))
                self.plainTextEdit.setText(report)           
                
    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        self.label.setText(_translate("Form", "Link Prediction in Social Network"))
        self.label_6.setText(_translate("Form", "Choose a dataset:"))
        self.label_4.setText(_translate("Form", "List dataset:"))
        self.comboBox.setItemText(0, _translate("Form", "Select Database"))
        self.comboBox.setItemText(1, _translate("Form", "USAir"))
        self.comboBox.setItemText(2, _translate("Form", "Lesmis"))
        self.comboBox.setItemText(3, _translate("Form", "NetSciences"))
        self.comboBox.setItemText(4, _translate("Form", "Rehall"))
        self.pushButton.setText(_translate("Form", "SAVE SETTING"))
        self.label_5.setText(_translate("Form", "Load dataset:"))
        self.label_15.setText(_translate("Form", "US Airports"))
        self.label_16.setText(_translate("Form", "<html><head/><body><p align=\"justify\">The directed network of flights beetween US Airports in 2010</p></body></html>"))
        self.label_17.setText(_translate("Form", "Les Miserables"))
        self.label_18.setText(_translate("Form", "<html><head/><body><p align=\"justify\">Coappearance network of characters in the novel Les Miserables.</p></body></html>"))
        self.label_34.setText(_translate("Form", "Net Sciences"))
        self.label_35.setText(_translate("Form", "<html><head/><body><p align=\"justify\">Coauthorship network of scientists working on network theory and experiment.</p><p align=\"justify\"><br/></p></body></html>"))
        self.label_36.setText(_translate("Form", "Rehall"))
        self.label_37.setText(_translate("Form", "<html><head/><body><p align=\"justify\">Friendship between residents living at a residence hall on the ANU campus.</p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("Form", "SELECT DATABASE"))
        self.label_38.setText(_translate("Form", "Choose type of Pre-processing:"))
        self.radioButton.setText(_translate("Form", "   Unweight Direct"))
        self.radioButton_2.setText(_translate("Form", "   Unweight Undirect"))
        self.radioButton_3.setText(_translate("Form", "   Weight Direct"))
        self.radioButton_4.setText(_translate("Form", "   Weight Undirect"))
        self.pushButton_3.setText(_translate("Form", "SAVE SETTING"))
        self.label_39.setText(_translate("Form", "<html><head/><body><p align=\"justify\">Get Adjacency matrix with unweight and direct link.</p><p align=\"justify\"><br/></p></body></html>"))
        self.label_40.setText(_translate("Form", "<html><head/><body><p align=\"justify\">Get Adjacency matrix with unweight and undirect link.</p><p align=\"justify\"><br/></p></body></html>"))
        self.label_41.setText(_translate("Form", "<html><head/><body><p align=\"justify\">Get Adjacency matrix with weight and direct link.</p><p align=\"justify\"><br/></p></body></html>"))
        self.label_65.setText(_translate("Form", "<html><head/><body><p align=\"justify\">Get Adjacency matrix with weight and undirect link.</p><p align=\"justify\"><br/></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "PRE-PROCESSING"))
        self.label_66.setText(_translate("Form", "Choose type of feature-Extractrion:"))
        self.pushButton_6.setText(_translate("Form", "SAVE SETTING"))
        self.checkBox.setText(_translate("Form", "   Common Neighbors"))
        self.checkBox_2.setText(_translate("Form", "   Jaccard"))
        self.checkBox_3.setText(_translate("Form", "   Adamic - Adar"))
        self.checkBox_4.setText(_translate("Form", "   Preferential Attachment"))
        self.checkBox_5.setText(_translate("Form", "   Reliable-route Common Neighbors"))
        self.checkBox_6.setText(_translate("Form", "   Reliable-route Jaccard"))
        self.checkBox_7.setText(_translate("Form", "   Reliable-route Adamic - Adar"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "FEATURE-EXTRACTION"))
        self.label_67.setText(_translate("Form", "Choose type of model Machine Learning:"))
        self.pushButton_7.setText(_translate("Form", "Analysis "))
        self.radioButton_9.setText(_translate("Form", "   SVM"))
        self.label_68.setText(_translate("Form", "Precision Score:"))
        self.radioButton_10.setText(_translate("Form", "   RBF - SVM"))
        self.radioButton_11.setText(_translate("Form", "   Deep Learning"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("Form", "MODEL ML"))
        self.label_7.setText(_translate("Form", "CORE TEAM"))
        self.label_8.setText(_translate("Form", "Duy Trinh Khanh Le"))
        self.label_9.setText(_translate("Form", "University of Information Technology - VNU HCM"))
        self.label_10.setText(_translate("Form", "15520159"))
        self.label_11.setText(_translate("Form", "Duong Huynh Cong Nguyen"))
        self.label_12.setText(_translate("Form", "15520148"))
        self.label_13.setText(_translate("Form", "Ty Ty Ta"))
        self.label_14.setText(_translate("Form", "15520996"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("Form", "INFORMATION"))
        self.label_3.setText(_translate("Form", "A new approach with Machine Learning"))
        #Add
        self.progressBar_1.reset()
        self.progressBar_4.reset()
        self.progressBar_2.reset()
        self.progressBar_3.reset()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

