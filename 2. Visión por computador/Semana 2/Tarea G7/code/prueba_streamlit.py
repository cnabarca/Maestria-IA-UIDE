import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


st.set_page_config(page_title="Clasificacion Cesar vs Fondo", layout="wide")

MODEL_CANDIDATES = [
	"modelo_clasificacion_cesar_fondo.h5",
	"modelo_cesar_fondo.keras",
]


def _find_model_path() -> str | None:
	base_dir = os.path.dirname(os.path.abspath(__file__))
	for file_name in MODEL_CANDIDATES:
		path = os.path.join(base_dir, file_name)
		if os.path.exists(path):
			return path
	return None


@st.cache_resource
def get_model(path: str):
	return load_model(path)


def predict_from_bgr(model, img_bgr: np.ndarray):
	img_resized = cv2.resize(img_bgr, (224, 224))
	img_batch = np.expand_dims(img_resized, axis=0)
	pred = model.predict(img_batch, verbose=0)
	p_fondo = float(pred[0][0])
	p_cesar = float(pred[0][1])
	label = "Fondo" if p_fondo > p_cesar else "Cesar"
	return label, p_fondo, p_cesar


def run_case(model, case_name: str, image_path: str, expected: str):
	img_bgr = cv2.imread(image_path)
	if img_bgr is None:
		return {
			"Caso": case_name,
			"Ruta": image_path,
			"Esperado": expected,
			"Predicho": "ERROR",
			"Resultado": "No se pudo leer la imagen",
		}, None

	pred_label, p_fondo, p_cesar = predict_from_bgr(model, img_bgr)
	row = {
		"Caso": case_name,
		"Ruta": image_path,
		"Esperado": expected,
		"Predicho": pred_label,
		"Resultado": "Correcto" if pred_label.lower() == expected.lower() else "Incorrecto",
		"Prob Fondo": round(p_fondo, 4),
		"Prob Cesar": round(p_cesar, 4),
	}
	return row, img_bgr


st.title("Clasificacion de Imagenes: Cesar vs Fondo")
st.write(
	"App de presentacion para validar las 3 pruebas del enunciado usando el modelo entrenado."
)

model_path = _find_model_path()
if model_path is None:
	st.error(
		"No se encontro el modelo en esta carpeta. Se esperaba uno de estos archivos: "
		+ ", ".join(MODEL_CANDIDATES)
	)
	st.stop()

st.success(f"Modelo cargado: {os.path.basename(model_path)}")
model = get_model(model_path)

tab1, tab2 = st.tabs(["Prediccion libre", "Pruebas del enunciado"])

with tab1:
	st.subheader("Prediccion con imagen subida")
	uploaded = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

	if uploaded is not None:
		file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
		img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
		if img_bgr is None:
			st.error("No se pudo procesar la imagen subida.")
		else:
			pred_label, p_fondo, p_cesar = predict_from_bgr(model, img_bgr)
			st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Imagen de entrada")
			st.markdown(f"### Prediccion: {pred_label}")
			st.write({"Prob Fondo": round(p_fondo, 4), "Prob Cesar": round(p_cesar, 4)})

with tab2:
	st.subheader("Configura y ejecuta las 3 pruebas")
	st.write("Revisa las rutas y presiona el boton para generar la tabla final.")

	default_base = r"C:/Users/cabarca2/MAESTRIA_IA/Vision-por-computador/semana 2/DEBER/data/pruebas"
	ruta1 = st.text_input(
		"Ruta prueba 1 (foto tuya)",
		value=f"{default_base}/prueba1_cesar6.jpg",
	)
	ruta2 = st.text_input(
		"Ruta prueba 2 (fondo no visto)",
		value=f"{default_base}/prueba2_fondo5.jpg",
	)
	ruta3 = st.text_input(
		"Ruta prueba 3 (otro rostro)",
		value=f"{default_base}/prueba3_otro_rostro1.jpg",
	)

	if st.button("Ejecutar 3 pruebas", type="primary"):
		cases = [
			("Prueba 1", ruta1, "Cesar"),
			("Prueba 2", ruta2, "Fondo"),
			("Prueba 3", ruta3, "Fondo"),
		]

		rows = []
		previews = []
		for case_name, image_path, expected in cases:
			row, img_bgr = run_case(model, case_name, image_path, expected)
			rows.append(row)
			previews.append((case_name, img_bgr))

		df = pd.DataFrame(rows)
		st.dataframe(df, use_container_width=True)
		csv_data = df.to_csv(index=False).encode("utf-8")
		st.download_button(
			label="Descargar resultados (CSV)",
			data=csv_data,
			file_name="resultados_3_pruebas.csv",
			mime="text/csv",
		)

		aciertos = int((df["Resultado"] == "Correcto").sum())
		st.markdown(f"### Aciertos: {aciertos}/3")

		cols = st.columns(3)
		for i, (case_name, img_bgr) in enumerate(previews):
			with cols[i]:
				st.caption(case_name)
				if img_bgr is None:
					st.warning("Sin vista previa")
				else:
					st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)