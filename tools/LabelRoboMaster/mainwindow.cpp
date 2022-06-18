#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QCoreApplication>
#include <iostream>
#include <QMessageBox>

class IndexQListWidgetItem : public QListWidgetItem {
public:
    IndexQListWidgetItem(QString name, int index) : QListWidgetItem(name), index(index) {

    }

    int getIndex() const { return index; }

private:
    int index;
};

MainWindow::MainWindow(QWidget *parent) :
        QMainWindow(parent),
        ui(new Ui::MainWindow) {
    ui->setupUi(this);
    QObject::connect(ui->addLabelPushButton, &QPushButton::clicked, ui->label, &DrawOnPic::setAddingMode);
    QObject::connect(ui->savePushButton, &QPushButton::clicked, ui->label, &DrawOnPic::saveLabel);
    QObject::connect(ui->stayPositionCheckBox, &QCheckBox::toggled, ui->label, &DrawOnPic::stayPositionChanged);
    ui->labelOpenvino->setText(ui->label->model_mode());  // 获取当前模型模式，并显示在窗口左下角
}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::on_openDirectoryPushButton_clicked() {
    ui->label->reset();     // 重置绘图控件
    QStringList image_filter = {"*.jpg", "*.png", "*.jpeg"};    // 支持的图像格式
    QDir dir = QFileDialog::getExistingDirectory(this, "", ".", QFileDialog::ShowDirsOnly);
    ui->fileListWidget->clear();    // 清空文件列表
    int idx = 0;
    // 遍历文件夹下的图片，并添加到文件列表
    for (QString file: dir.entryList(image_filter)) {
        if (file == "." || file == "..") continue;
        ui->fileListWidget->addItem(new IndexQListWidgetItem(dir.absoluteFilePath(file), idx++));
    }
    ui->fileListWidget->setCurrentItem(ui->fileListWidget->item(0));
    // 设置拖动条
    ui->fileListHorizontalSlider->setMinimum(1);
    ui->fileListHorizontalSlider->setMaximum(ui->fileListWidget->count());
    ui->fileListHorizontalSlider->setValue(1);
}

void MainWindow::on_fileListWidget_currentItemChanged(QListWidgetItem *current, QListWidgetItem *previous) {
    if (current == nullptr) return;
    // 修改绘图区的内容
    ui->label->setCurrentFile(current->text());
    // 更新拖动条位置
    int idx = static_cast<IndexQListWidgetItem *>(current)->getIndex();
    ui->fileListHorizontalSlider->setValue(idx + 1);
}

void MainWindow::on_label_labelChanged(const QVector<box_t> &labels) {
    // 当添加/删除/修改当前图片的label时，重置右上角的标签列表
    // TODO: 可以有性能优化
    ui->labelListWidget->clear();
    for (int i = 0; i < labels.size(); i++) {
        ui->labelListWidget->addItem(new IndexQListWidgetItem(labels[i].getName(), i));
    }
}

void MainWindow::on_labelListWidget_itemDoubleClicked(QListWidgetItem *item) {
    // 双击打开label设置窗口（修改label类别/删除label）
    int idx = static_cast<IndexQListWidgetItem *>(item)->getIndex();
    delete dialog;
    dialog = new LabelDialog(ui->label->get_current_label().begin() + idx);
    dialog->setModal(true);
    QObject::connect(dialog, &LabelDialog::removeBoxEvent, ui->label, &DrawOnPic::removeBox);
    QObject::connect(dialog, &LabelDialog::changeBoxEvent, ui->label, &DrawOnPic::updateBox);
    dialog->show();
}

void MainWindow::on_labelListWidget_currentItemChanged(QListWidgetItem *current, QListWidgetItem *previous) {
    // 更新当前选中目标（绘图区域目标高亮）
    if (current == nullptr) return;
    int idx = static_cast<IndexQListWidgetItem *>(current)->getIndex();
    ui->label->setFocusBox(idx);
}

void MainWindow::on_smartPushButton_clicked() {
    // 进行一次自动标注
    ui->label->smart();
}

void MainWindow::on_smartAllPushButton_clicked() {
    // 遍历所有目标，依次进行自动标注
    // TODO: 目前无法中断该过程
    for (int i = 0; i < ui->fileListWidget->count(); i++) {
        ui->fileListWidget->setCurrentRow(i);
        ui->label->smart();
        // ui->label->setCurrentFile(ui->fileListWidget->item(i)->text());
        ui->label->saveLabel();
        QCoreApplication::processEvents();
    }
}

void MainWindow::on_nextPushButton_clicked() {
    // 切换图片时，判断是否需要自动保存
    if (ui->autoSaveCheckBox->checkState() == Qt::Checked) {
        ui->label->saveLabel();
    }
    // 更新左下角文件列表中的选中项（通过信号更新绘图区域内容）
    int next_idx = ui->fileListWidget->currentRow() + 1;
    if (next_idx < ui->fileListWidget->count()) {
        ui->fileListWidget->setCurrentRow(next_idx);
    }
}

// 切换为上一个目标。和切换下一个目标类似。
void MainWindow::on_prevPushButton_clicked() {
    if (ui->autoSaveCheckBox->checkState() == Qt::Checked) {
        ui->label->saveLabel();
    }
    int prev_idx = ui->fileListWidget->currentRow() - 1;
    if (prev_idx >= 0) {
        ui->fileListWidget->setCurrentRow(prev_idx);
    }
}

void MainWindow::on_fileListHorizontalSlider_valueChanged(int value) {
    // 响应文件列表拖动条
    QString text;
    ui->fileListLabel->setText(text.sprintf("[%d/%d]", value, ui->fileListHorizontalSlider->maximum()));
    ui->fileListWidget->setCurrentItem(ui->fileListWidget->item(value - 1));
}

void MainWindow::on_fileListHorizontalSlider_rangeChanged(int min, int max) {
    // 响应文件列表拖动条
    QString text;
    ui->fileListLabel->setText(text.sprintf("[%d/%d]", ui->fileListHorizontalSlider->value(), max));
}

void MainWindow::on_interpolateButton_clicked() {
    enum InterpolateStatus {
        Idle, Waiting4A, Waiting4B
    };// 三种状态：闲置，准备选择第一个插值点，准备选择第二个插值点
    static InterpolateStatus status = Idle;
    static int target_idx = 0;
    static box_t box_a, box_b;
    switch (status) {
        case Idle:
            target_idx = ui->fileListWidget->currentRow();
//            if (target_idx - 1 < 0 || target_idx + 1 >= ui->fileListWidget->count()) { //考虑到后面实际上可以随意选择插值来源，这个判断可有可无
//                QMessageBox::warning(this, "Invalid interpolate", "Can't interpolate boxes on first or last picture.");
//            } else {
            if (ui->autoSaveCheckBox->isChecked()) ui->label->saveLabel();
            ui->fileListWidget->setCurrentRow(target_idx - 1); // 自动跳转到上一张图
            status = Waiting4A;
//            }
            break;
        case Waiting4A:
        case Waiting4B: {
            auto selected = ui->labelListWidget->selectedItems();
            if (selected.empty()) {
                QMessageBox::warning(this, "No boxes selected", "Select a box first");
            } else {
                if (status == Waiting4A) { // 读取到box_a
                    box_a = ui->label->get_current_label().at(
                            (dynamic_cast<IndexQListWidgetItem *>(selected.first()))->getIndex());
                    ui->fileListWidget->setCurrentRow(target_idx + 1); // 自动跳转到下一张
                    status = Waiting4B;
                } else {
                    box_b = ui->label->get_current_label().at(
                            (dynamic_cast<IndexQListWidgetItem *>(selected.first()))->getIndex());
                    if (box_a.tag_id == box_b.tag_id) { // 检测装甲板类型是否相同
                        box_t result;
                        result.tag_id = box_a.tag_id;
                        result.color_id = box_a.color_id;
                        for (int i = 0; i < 4; ++i) result.pts[i] = box_a.pts[i] / 2 + box_b.pts[i] / 2; // 插值
                        ui->fileListWidget->setCurrentRow(target_idx);
                        ui->label->get_current_label().append(result);
                        ui->label->updateBox();
                    } else {
                        QMessageBox::warning(this, "Shouldn't interpolate boxes of different types",
                                             "Please select boxes of same type");
                        ui->fileListWidget->setCurrentRow(target_idx);
                    }
                    status = Idle;
                }
            }
        }
            break;
    }
}

void MainWindow::on_upLabelButton_clicked() {
    int prev_idx = ui->labelListWidget->currentRow() - 1;
    ui->labelListWidget->setCurrentRow(prev_idx >= 0 ? prev_idx : ui->labelListWidget->count() - 1);
}

void MainWindow::on_downLabelButton_clicked() {
    int next_idx = ui->labelListWidget->currentRow() + 1;
    ui->labelListWidget->setCurrentRow(next_idx < ui->labelListWidget->count() ? next_idx : 0);
}
