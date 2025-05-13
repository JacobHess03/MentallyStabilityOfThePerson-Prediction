import os
import sys
import threading
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox,
    QLineEdit, QTextEdit, QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from preprocessing import preprocess_train, elimina_variabili_vif_pvalue, preprocess_person_test
import graphicGui

# Costanti
CLEANED_CSV = 'cleaned_data_for_graphs.csv'
DEFAULT_MODEL_FILE = 'best_xgb_clf_smote.pkl'


class Worker(QObject):
    status = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def run(self):
        try:
            self.status.emit("Caricamento dati...")
            df = pd.read_csv(self.data_path)
            self.status.emit("Preprocessing...")
            df_clean = preprocess_train(df)
            df_clean.to_csv(CLEANED_CSV, index=False)

            X = df_clean.drop(columns=['Depression'])
            y = df_clean['Depression']
            self.status.emit("Selezione variabili...")
            X_sel = elimina_variabili_vif_pvalue(X, y)

            self.status.emit("Preparazione modelli...")
            neg, pos = np.bincount(y)
            scale = neg / pos
            X_train, X_test, y_train, y_test = train_test_split(
                X_sel, y, test_size=0.2, random_state=73, stratify=y
            )

            log_reg = LogisticRegression(random_state=73, max_iter=1000, class_weight='balanced')
            xgb = XGBClassifier(objective='binary:logistic', random_state=73, scale_pos_weight=scale)
            smote = SMOTE(random_state=73)
            pipeline = ImbPipeline([
                ('smote', smote),
                ('xgb', XGBClassifier(objective='binary:logistic', random_state=73))
            ])
            grid = GridSearchCV(pipeline, {
                'xgb__n_estimators': [50, 100],
                'xgb__max_depth': [3, 5],
                'xgb__learning_rate': [0.05, 0.1],
                'xgb__scale_pos_weight': [1, scale]
            }, scoring='f1', cv=5, n_jobs=-1)

            self.status.emit("Addestramento modelli...")
            log_reg.fit(X_train, y_train)
            xgb.fit(X_train, y_train)
            grid.fit(X_train, y_train)

            best = grid.best_estimator_
            joblib.dump(best, DEFAULT_MODEL_FILE)
            joblib.dump(log_reg, 'logistic_regression_smote.pkl')
            joblib.dump(xgb, 'xgb_clf_default_smote.pkl')

            self.status.emit("Addestramento completato.")
        except Exception as e:
            self.status.emit(f"Errore: {e}")
        finally:
            self.finished.emit()


class TrainPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        form = QHBoxLayout()
        self.path_edit = QLineEdit()
        btn_browse = QPushButton("Sfoglia...")
        form.addWidget(QLabel("Percorso CSV:"))
        form.addWidget(self.path_edit)
        form.addWidget(btn_browse)
        layout.addLayout(form)

        self.start_btn = QPushButton("Avvia Addestramento")
        layout.addWidget(self.start_btn)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        # self.back_btn = QPushButton("Indietro")
        # layout.addWidget(self.back_btn)

        btn_browse.clicked.connect(self.browse)
        self.start_btn.clicked.connect(self.start)

    def browse(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Seleziona CSV", ".", "CSV files (*.csv)")
        if fp:
            self.path_edit.setText(fp)

    def start(self):
        path = self.path_edit.text().strip()
        if not os.path.exists(path):
            QMessageBox.warning(self, "Errore", "File non esistente")
            return

        self.log.clear()
        self.start_btn.setEnabled(False)

        self.worker = Worker(path)
        thread = threading.Thread(target=self.worker.run)
        self.worker.status.connect(self.log.append)
        self.worker.finished.connect(lambda: self.start_btn.setEnabled(True))
        thread.start()


class GraphsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Top: select CSV
        top = QHBoxLayout()
        self.lbl_datafile = QLabel(f"Dati: {CLEANED_CSV}")
        btn_load_csv = QPushButton("Carica Dati…")
        top.addWidget(self.lbl_datafile)
        top.addStretch()
        top.addWidget(btn_load_csv)
        layout.addLayout(top)

        # Body: buttons + canvas
        body = QHBoxLayout()
        self.btn_layout = QVBoxLayout()
        self.canvas_holder = QVBoxLayout()
        body.addLayout(self.btn_layout)
        body.addLayout(self.canvas_holder)
        layout.addLayout(body)

        # self.back_btn = QPushButton("Indietro")
        # layout.addWidget(self.back_btn, alignment=Qt.AlignRight)

        self.data_file = CLEANED_CSV
        btn_load_csv.clicked.connect(self.browse_csv)

        self.graph_funcs = {
            "Distribuzione Genere":      graphicGui.plot_gender_distribution,
            "Studente vs Professionista":graphicGui.plot_status_distribution,
            "Condizione di Depressione": graphicGui.plot_depression_distribution,
            "Depressione per Genere":    graphicGui.plot_depression_by_gender,
            "Stress Finanziario":        graphicGui.plot_financial_stress_by_depression,
            "Età vs Depress.":           graphicGui.plot_age_distribution_by_depression,
            "Correlaz. Pearson":         graphicGui.plot_pearson_correlation,
            "Depress. per Laurea":       graphicGui.plot_depression_by_Degree_Group_Encoded_group,
            "Soddisf. Studio":           graphicGui.plot_depression_by_study_satisfaction,
            "Depress. per Status":       graphicGui.plot_depression_by_status
        }

        for name in self.graph_funcs:
            btn = QPushButton(name)
            self.btn_layout.addWidget(btn)
           
            
            btn.clicked.connect(lambda _, name=name: self.show_graph(name))
           

    def browse_csv(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Seleziona CSV pulito", ".", "CSV files (*.csv)")
        if fp:
            self.data_file = fp
            self.lbl_datafile.setText(f"Dati: {os.path.basename(fp)}")

    def load_data(self):
        if not os.path.exists(self.data_file):
            return None
        try:
            return pd.read_csv(self.data_file)
        except:
            return None

    def show_graph(self, name):
        
        df = self.load_data()
        if df is None:
            QMessageBox.warning(self, "Errore", "Dati non disponibili. Carica prima un CSV.")
            return

        fig = self.graph_funcs[name](df)

        # remove old widget
        for i in reversed(range(self.canvas_holder.count())):
            w = self.canvas_holder.itemAt(i).widget()
            if w:
                w.setParent(None)

        canvas = FigureCanvas(fig)
        self.canvas_holder.addWidget(canvas)


class PredictPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # Model loader
        hmod = QHBoxLayout()
        self.lbl_model = QLabel(f"Modello: {os.path.basename(DEFAULT_MODEL_FILE)}")
        btn_load = QPushButton("Carica Modello…")
        hmod.addWidget(self.lbl_model)
        hmod.addStretch()
        hmod.addWidget(btn_load)
        layout.addLayout(hmod)

        self.model = None
        self.load_model(DEFAULT_MODEL_FILE)
        btn_load.clicked.connect(self.browse_model)

        # Features form
        self.features = {
            'Name': {'type':'text'},
            'Gender': {'type':'categorical','options':['Male','Female']},
            'Age': {'type':'numeric','range':(18,60)},
            'City': {'type':'text'},
            'Working Professional or Student':{'type':'text'},
            'Profession':{'type':'text'},
            'Academic Pressure':{'type':'numeric','range':(0,5)},
            'Work Pressure':{'type':'numeric','range':(0,5)},
            'CGPA':{'type':'numeric','range':(0.0,10.0)},
            'Study Satisfaction':{'type':'numeric','range':(0,5)},
            'Job Satisfaction':{'type':'numeric','range':(0,5)},
            'Sleep Duration':{'type':'numeric','range':(0.0,12.0)},
            'Dietary Habits':{'type':'categorical','options':['Unhealthy','Moderate','Healthy']},
            'Degree':{'type':'text'},
            'Have you ever had suicidal thoughts ?':{'type':'categorical','options':['Yes','No']},
            'Work/Study Hours':{'type':'numeric','range':(0.0,12.0)},
            'Financial Stress':{'type':'numeric','range':(0,5)},
            'Family History of Mental Illness':{'type':'categorical','options':['Yes','No']}
        }
        form = QFormLayout()
        self.inputs = {}
        for col, defn in self.features.items():
            if defn['type']=='categorical':
                cb = QComboBox()
                cb.addItems(defn['options'])
                form.addRow(col, cb)
                self.inputs[col] = cb
            else:
                le = QLineEdit()
                label = f"{col} ({defn['range'][0]}-{defn['range'][1]})" if defn['type']=='numeric' else col
                form.addRow(label, le)
                self.inputs[col] = le
        layout.addLayout(form)

        self.btn_pred = QPushButton("Esegui Predizione")
        self.lbl_res  = QLabel("Inserisci i dati e premi Predizione")
        # btn_back      = QPushButton("Indietro")
        layout.addWidget(self.btn_pred)
        layout.addWidget(self.lbl_res)
        # layout.addWidget(btn_back)

        self.btn_pred.clicked.connect(self.perform_prediction)
        # btn_back.clicked.connect(lambda: self.parent().setCurrentIndex(0))

    def browse_model(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Seleziona modello .pkl", ".", "Pickle files (*.pkl)")
        if fp:
            try:
                self.model = joblib.load(fp)
                self.lbl_model.setText(f"Modello: {os.path.basename(fp)}")
                QMessageBox.information(self, "Modello Caricato", f"Caricato: {os.path.basename(fp)}")
            except Exception as e:
                QMessageBox.critical(self, "Errore Caricamento", str(e))

    def load_model(self, path):
        try:
            self.model = joblib.load(path)
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile caricare default: {e}")

    def perform_prediction(self):
        data = {'id':[0]}
        errors = []
        for col, defn in self.features.items():
            widget = self.inputs[col]
            raw = widget.currentText() if defn['type']=='categorical' else widget.text().strip()
            if defn['type']=='numeric':
                if not raw:
                    errors.append(f"'{col}' mancante")
                    continue
                try:
                    val = float(raw) if isinstance(defn['range'][0], float) else int(raw)
                except:
                    errors.append(f"'{col}' non valido")
                    continue
                data[col] = [val]
            else:
                data[col] = [raw]
        if errors:
            QMessageBox.warning(self, "Errori di input", "\n".join(errors))
            return
        df = pd.DataFrame(data)
        try:
            df_clean = preprocess_person_test(df)
            y = self.model.predict(df_clean)
            if hasattr(self.model, 'predict_proba'):
                p = self.model.predict_proba(df_clean)[:,1][0]
                text = f"Depressione: {'SI' if y[0]==1 else 'NO'} (Prob: {p:.2%})"
            else:
                text = f"Depressione: {'SI' if y[0]==1 else 'NO'}"
            self.lbl_res.setText(text)
        except Exception as e:
            QMessageBox.critical(self, "Errore Predizione", str(e))


class DepressionAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analisi e Predizione Depressione")
        self.resize(900, 700)

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Barra di navigazione in alto
        nav_bar = QWidget()
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(5, 5, 5, 5)
        self.btn_train  = QPushButton("Addestra Modelli")
        self.btn_graphs = QPushButton("Visualizza Grafici")
        self.btn_predict= QPushButton("Predizione Singola")
        nav_layout.addWidget(self.btn_train)
        nav_layout.addWidget(self.btn_graphs)
        nav_layout.addWidget(self.btn_predict)
        nav_layout.addStretch()
        main_layout.addWidget(nav_bar)

        # Stack delle pagine
        self.stack = QStackedWidget()
        self.train_page   = TrainPage(self)
        self.graphs_page  = GraphsPage(self)
        self.predict_page = PredictPage(self)
        self.stack.addWidget(self.train_page)
        self.stack.addWidget(self.graphs_page)
        self.stack.addWidget(self.predict_page)
        main_layout.addWidget(self.stack)

        # Connect navigation
        self.btn_train.clicked.connect(lambda: self.stack.setCurrentWidget(self.train_page))
        self.btn_graphs.clicked.connect(lambda: self.stack.setCurrentWidget(self.graphs_page))
        self.btn_predict.clicked.connect(lambda: self.stack.setCurrentWidget(self.predict_page))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DepressionAnalysisApp()
    window.show()
    sys.exit(app.exec_())
