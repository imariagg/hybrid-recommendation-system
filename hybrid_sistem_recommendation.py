import pandas as pd
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np
import json
from psycopg2.extras import DictCursor
from sklearn.model_selection import ParameterGrid

# Clase para la conexión a la base de datos y la extracción de datos
class ConexionBaseDeDatos:
    """
    Gestiona la conexión y las consultas a una base de datos PostgreSQL.
    Carga la configuración desde un archivo JSON para la conexión, facilitando la gestión
    de la conexión y la ejecución de consultas SQL.
    """
    
    def __init__(self):
        """
        Inicializa una nueva instancia de ConexionBaseDeDatos.
        Carga la configuración desde 'config.json', estableciendo los parámetros necesarios para la conexión.
        """
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
            db_config = config['database']

        self.host = db_config['host']
        self.base_de_datos = db_config['database_name']
        self.puerto = db_config['port']
        self.usuario = db_config['username']
        self.contraseña = db_config['password']
        self.conexion = None
        self.cursor = None

    def conectar(self):
        try:
            if self.conexion is None or self.conexion.closed:
                with psycopg2.connect(
                    host=self.host,
                    port=self.puerto,
                    database=self.base_de_datos,
                    user=self.usuario,
                    password=self.contraseña
                ) as conexion:
                    self.cursor = conexion.cursor(cursor_factory=DictCursor)
                self.conexion = conexion
        except psycopg2.Error as e:
            print(f"Error al conectar a la base de datos: {e}")
            self.conexion = None
            self.cursor = None


    def fetch_data(self, intervenciones_no_recomendadas):
        self.conectar()
        if self.conexion is None:
            print("No se pudo conectar a la base de datos.")
            return None, None, None, None

        try:
            with self.conexion.cursor(cursor_factory=DictCursor) as cursor:
               
                # Lista de IDs de intervenciones no recomendadas
                ids_no_recomendados = tuple(intervenciones_no_recomendadas)

                # Consulta de usuarios (no necesita filtro)
                cursor.execute('SELECT "Id", "EstadoMaternidad", "PrimerHijo", "ParejaActual", "DepresionOAnsiedadPrevia", "ComplicacionesEmbarazo", "PerdidaBebe", "EpdsGad7AmarilloONaranja", "ContraindicacionesEjercicio" FROM "AspNetUsers";')
                users = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

                # Consulta de intervenciones, excluyendo las no recomendadas
                cursor.execute(f'''
                    SELECT "IdIntervencion", "TituloApp", "Titulo", "Formato", "Creador", "AsignacionProfesional"
                    FROM "Intervenciones"
                    WHERE "IdIntervencion" NOT IN {ids_no_recomendados};
                ''')
                intervenciones = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])


                # Consulta de tags, excluyendo intervenciones no recomendadas
                cursor.execute(f'''
                    SELECT "IdIntervencion", "Tag1", "Tag2", "Tag3", "Tag4", "Tag5"
                    FROM "Intervencion_Tag"
                    WHERE "IdIntervencion" NOT IN {ids_no_recomendados};
                ''')
                tags = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

                # Consulta de interacciones, excluyendo intervenciones no recomendadas
                cursor.execute(f'''
                    SELECT "IdUser", "IdIntervencion", "Porc_Completado", "valoracion", "Guardado", "Marcado"
                    FROM "Usuario_Intervencion"
                    WHERE "IdIntervencion" NOT IN {ids_no_recomendados};
                ''')
                interacciones = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
                
            return users, intervenciones, tags, interacciones
        except psycopg2.Error as e:
            print(f"Error durante la ejecución de la consulta: {e}")
            return None, None, None, None
        finally:
            self.conexion.close()


    def desconectar(self):
        if self.conexion:
            self.conexion.close()


class Usuario:
    """
    Clase para gestionar la información y condiciones de un usuario.

    Atributos:
        id_usuario (str): ID del usuario.
        conexion_db (ConexionBaseDeDatos): Instancia de la conexión a la base de datos.
        username (str): Nombre de usuario.
        sex (str): Género del usuario.
        estado_maternidad (int): Estado de maternidad del usuario.
        primer_hijo (bool): Indica si es el primer hijo.
        id_acompanante (str): ID del acompañante.
        pareja_actual (str): Pareja actual del usuario.
        depresion_o_ansiedad_previa (bool): Indica si el usuario tiene antecedentes de depresión o ansiedad.
        complicaciones_embarazo (bool): Indica si hubo complicaciones durante el embarazo.
        perdida_bebe (bool): Indica si hubo pérdida del bebé.
        epds_gad_7_amarillo_o_naranja (bool): Indica si hay una alerta de EPDS o GAD 7 en amarillo o naranja.
        fecha_parto (datetime): Fecha del parto.
        fecha_ultima_regla (datetime): Fecha de la última regla.
        contraindicaciones_ejercicio (bool): Indica si hay contraindicaciones para el ejercicio.
        semana_perinatal (str): Semana perinatal calculada.
        intervenciones_no_recomendadas (list): Lista de intervenciones no recomendadas.
    """

    def __init__(self, id_usuario, conexion_db):
        """
        Inicializa un nuevo usuario y carga sus datos desde la base de datos.

        Args:
            id_usuario (str): ID del usuario.
            conexion_db (ConexionBaseDeDatos): Instancia de la conexión a la base de datos.
        """
        self.id_usuario = id_usuario
        self.conexion_db = conexion_db
        self.username = None
        self.sex = None
        self.estado_maternidad = 0
        self.primer_hijo = None
        self.id_acompanante = None
        self.pareja_actual = None
        self.depresion_o_ansiedad_previa = None
        self.complicaciones_embarazo = None
        self.perdida_bebe = None
        self.epds_gad_7_amarillo_o_naranja = None
        self.fecha_parto = None
        self.fecha_ultima_regla = None
        self.contraindicaciones_ejercicio = None
        self.cargar_datos_usuario()
        self.mama_con_pareja_mujer = self.calcular_mama_con_pareja('mujer')
        self.mama_con_pareja_hombre = self.calcular_mama_con_pareja('hombre')
        self.mama_pareja = self.estado_maternidad in [2, 3] and self.sex == 'mujer'
        self.solo_pareja = self.estado_maternidad in [2, 3] and self.sex == 'hombre'
        self.mama_sin_pareja = self.estado_maternidad in [0, 1]
        self.intervenciones_no_recomendadas = self.obtener_intervenciones_no_recomendadas()
        print(self.intervenciones_no_recomendadas)
        self.obtener_intervenciones_completadas() 
        

    def cargar_datos_usuario(self):
        """
        Carga los datos del usuario desde la base de datos.
        """
        query = """
        SELECT "UserName", "Sex", "EstadoMaternidad", "PrimerHijo", "IdAcompañante", "ParejaActual", 
               "DepresionOAnsiedadPrevia", "ComplicacionesEmbarazo", "PerdidaBebe", "EpdsGad7AmarilloONaranja", 
               "FechaParto", "FechaUltimaRegla", "ContraindicacionesEjercicio"
        FROM "AspNetUsers"
        WHERE "Id" = %s;
        """
        try:
            self.conexion_db.conectar()
            cursor = self.conexion_db.cursor
            cursor.execute(query, (self.id_usuario,))
            resultado = cursor.fetchone()

            if resultado:
                self.username = resultado['UserName']
                self.sex = resultado['Sex']
                self.estado_maternidad = resultado['EstadoMaternidad']
                self.primer_hijo = resultado['PrimerHijo']
                self.id_acompanante = resultado['IdAcompañante']
                self.pareja_actual = resultado['ParejaActual']
                self.depresion_o_ansiedad_previa = resultado['DepresionOAnsiedadPrevia']
                self.complicaciones_embarazo = resultado['ComplicacionesEmbarazo']
                self.perdida_bebe = resultado['PerdidaBebe']
                self.epds_gad_7_amarillo_o_naranja = resultado['EpdsGad7AmarilloONaranja']
                self.fecha_parto = resultado['FechaParto']
                self.fecha_ultima_regla = resultado['FechaUltimaRegla']
                self.contraindicaciones_ejercicio = resultado['ContraindicacionesEjercicio']
            else:
                print("No se encontró el usuario con el ID proporcionado o hubo un error en la consulta.")

        except Exception as e:
            print(f"Error al cargar los datos del usuario: {e}")

        finally:
            self.conexion_db.desconectar()


    def calcular_mama_con_pareja(self, sexo_pareja):
        """
        Calcula si la madre tiene pareja con el sexo especificado.

        Args:
            sexo_pareja (str): Sexo de la pareja ('hombre' o 'mujer').

        Returns:
            bool: True si la madre tiene pareja con el sexo especificado, False en caso contrario.
        """
        if self.estado_maternidad in [2, 3] and self.sex == sexo_pareja:
            return True
        elif self.estado_maternidad in [0, 1] and self.id_acompanante:
            query = """
            SELECT "Sex"
            FROM "AspNetUsers"
            WHERE "Id" = %s;
            """
            try:
                self.conexion_db.conectar()
                cursor = self.conexion_db.cursor
                cursor.execute(query, (self.id_acompanante,))
                resultado = cursor.fetchone()
                return resultado and resultado['Sex'] == sexo_pareja
            except Exception as e:
                print(f"Error al determinar si la madre tiene pareja {sexo_pareja}: {e}")
            finally:
                self.conexion_db.desconectar()
        return False


    def calcular_semana_perinatal(self):
        """
        Calcula la semana perinatal en función de la fecha de última regla o la fecha de parto.

        Returns:
            str: Semana perinatal calculada ('pre_wX' o 'post_wX'), o None si no se puede calcular.
        """
        if self.estado_maternidad in [0, 2] and self.fecha_ultima_regla:
            semanas = (datetime.datetime.now().date() - self.fecha_ultima_regla.date()).days // 7
            return f"pre_w{semanas}"
        elif self.estado_maternidad in [1, 3] and self.fecha_parto:
            semanas = (datetime.datetime.now().date() - self.fecha_parto.date()).days // 7
            return f"post_w{semanas}"
        return None

    def obtener_intervenciones_no_recomendadas(self):
        """
        Obtiene las intervenciones no recomendadas según las características del usuario.
        Asegura que las características necesarias no sean None antes de realizar la consulta.
        Returns:
            list: Lista de IDs de intervenciones no recomendadas.
        """


        columnas_a_consultar = {
            "MamaPrimeriza": self.primer_hijo,
            "MamaMultiPara": not self.primer_hijo,
            "MamaConParejaMujer": self.mama_con_pareja_mujer,
            "MamaSinPareja": self.mama_sin_pareja,
            "MamaPareja": self.mama_pareja,
            "SoloPareja": self.solo_pareja,
            "MamaConParejaHombre": self.mama_con_pareja_hombre,
            "DepresionAnsiedadPreviaSi": self.depresion_o_ansiedad_previa,
            "DepresionAnsiedadPreviaNo": not self.depresion_o_ansiedad_previa,
            "ComplicacionesEmbarazoSi": self.complicaciones_embarazo,
            "ComplicacionesEmbarazoNo": not self.complicaciones_embarazo,
            "PerdidaBebeSi": self.perdida_bebe,
            "EpdsGad7AmarilloNaranja": self.epds_gad_7_amarillo_o_naranja,
            "ContraindicacionesEjercicio": self.contraindicaciones_ejercicio
        }

        columnas_seleccionadas = [columna for columna, condicion in columnas_a_consultar.items() if condicion]
       

        if not columnas_seleccionadas:
            print("No hay características del usuario que apliquen para esta consulta.")
            return []

        consulta = f"""
        SELECT "IdIntervencion"
        FROM "Intervenciones"
        WHERE {" OR ".join([f'"{columna}" = -1' for columna in columnas_seleccionadas])};
        """

        try:
            self.conexion_db.conectar()
            self.conexion_db.cursor.execute(consulta)
            resultados = self.conexion_db.cursor.fetchall()
            return [resultado["IdIntervencion"] for resultado in resultados] if resultados else []
        except Exception as e:
            print(f"Error al obtener las intervenciones no recomendadas: {e}")
            return []
        finally:
            self.conexion_db.desconectar()

    def obtener_intervenciones_completadas(self):
        """
        Identifica las intervenciones que el usuario ya ha completado (más del 85%) o ha valorado.
        """
        query = """
        SELECT "IdIntervencion"
        FROM "Usuario_Intervencion"
        WHERE "IdUser" = %s AND ("Porc_Completado" >= 85 OR "valoracion" IS NOT NULL);
        """
        try:
            self.conexion_db.conectar()
            cursor = self.conexion_db.cursor
            cursor.execute(query, (self.id_usuario,))
            resultados = cursor.fetchall()
            self.intervenciones_completadas = [resultado["IdIntervencion"] for resultado in resultados] if resultados else []
        except Exception as e:
            print(f"Error al obtener las intervenciones completadas: {e}")
        finally:
            self.conexion_db.desconectar()

    def print_usuario(self):
        """
        Imprime los detalles del usuario en formato legible.
        """
        print("Detalles del Usuario:")
        print(f"ID: {self.id_usuario}")
        print(f"Nombre de usuario: {self.username}")
        print(f"Género: {self.sex}")
        print(f"Estado de maternidad: {self.estado_maternidad}")
        print(f"Es su primer hijo: {self.primer_hijo}")
        print(f"ID del acompañante: {self.id_acompanante}")
        print(f"Pareja actual: {self.pareja_actual}")
        print(f"Depresión o ansiedad previa: {self.depresion_o_ansiedad_previa}")
        print(f"Complicaciones en el embarazo: {self.complicaciones_embarazo}")
        print(f"Ha tenido un aborto espontáneo: {self.perdida_bebe}")
        print(f"Alerta EPDS GAD 7 amarilla o naranja: {self.epds_gad_7_amarillo_o_naranja}")
        print(f"Fecha de parto: {self.fecha_parto}")
        print(f"Fecha de última regla: {self.fecha_ultima_regla}")
        print(f"Contraindicaciones de ejercicio: {self.contraindicaciones_ejercicio}")
        print(f"Mamá con pareja mujer: {self.mama_con_pareja_mujer}")
        print(f"Mamá con pareja hombre: {self.mama_con_pareja_hombre}")
        print(f"Mamá con pareja: {self.mama_pareja}")
        print(f"Solo pareja: {self.solo_pareja}")
        print(f"Mamá sin pareja: {self.mama_sin_pareja}")
        print(f"Intervenciones que no puede realizar: {self.intervenciones_no_recomendadas}")

# Clase para la preparación de los datos
class DataPreprocessor:
    def __init__(self, users, intervenciones, tags, interacciones):
        self.users = users
        self.intervenciones = intervenciones
        self.tags = tags
        self.interacciones = interacciones

    def preprocess(self):
        self.interacciones = self.interacciones[self.interacciones['IdIntervencion'] > 0]
        self.interacciones = self.interacciones[self.interacciones['IdUser'] != 0]

        # Combinar los tags en una sola columna y añadir más características
        self.tags['Tags'] = self.tags[['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5']].fillna('').agg(' '.join, axis=1).str.strip()
        intervenciones_con_tags = self.intervenciones.merge(self.tags[['IdIntervencion', 'Tags']], on='IdIntervencion', how='left')

        # Combinar metadatos
        intervenciones_con_tags['Combined'] = (
            intervenciones_con_tags['TituloApp'].fillna('') + ' ' +
            intervenciones_con_tags['Titulo'].fillna('') + ' ' +
            intervenciones_con_tags['Formato'].fillna('') + ' ' +
            intervenciones_con_tags['Creador'].fillna('') + ' ' +
            intervenciones_con_tags['AsignacionProfesional'].fillna('') + ' ' +
            intervenciones_con_tags['Tags'].fillna('')
        )

        # Matriz de interacciones
        interactions_matrix = self.interacciones.pivot(index='IdUser', columns='IdIntervencion', values='valoracion').fillna(0)

        # Ajuste de valoraciones
        for index, row in self.interacciones.iterrows():
            if pd.isnull(row['valoracion']):
                completado_weight = max(0.8, row['Porc_Completado'] / 100)
                guardado_weight = 1.2 if row['Guardado'] else 1.0
                marcado_weight = 1.4 if row['Marcado'] else 1.0
                valoracion_implicita = completado_weight * guardado_weight * marcado_weight * 5
                interactions_matrix.loc[row['IdUser'], row['IdIntervencion']] = valoracion_implicita
            else:
                completado_weight = max(0.8, row['Porc_Completado'] / 100)
                guardado_weight = 1.2 if row['Guardado'] else 1.0
                marcado_weight = 1.4 if row['Marcado'] else 1.0
                valoracion_ajustada = float(row['valoracion']) * completado_weight * guardado_weight * marcado_weight
                interactions_matrix.loc[row['IdUser'], row['IdIntervencion']] = min(valoracion_ajustada, 6)

        return interactions_matrix, intervenciones_con_tags

class ContentBasedRecommender:
    def __init__(self, intervenciones):
        self.intervenciones = intervenciones
        self.vectorizer = TfidfVectorizer(max_features=10000, min_df=1, max_df=0.9, ngram_range=(1,2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.intervenciones['Combined'])

    def build_user_profile(self, user_id, interactions_matrix):
        user_ratings = interactions_matrix.loc[user_id]
        user_ratings = user_ratings[user_ratings > 0]

        if user_ratings.empty:
            return None

        user_profile = np.zeros(self.tfidf_matrix.shape[1])
        for intervencion_id, rating in user_ratings.items():
            idx = self.intervenciones[self.intervenciones['IdIntervencion'] == intervencion_id].index[0]
            user_profile += self.tfidf_matrix[idx].toarray().flatten() * rating

        user_profile = user_profile / user_ratings.sum()
        return user_profile

    def recommend(self, user_id, interactions_matrix, top_n=5):
        user_profile = self.build_user_profile(user_id, interactions_matrix)
        if user_profile is None:
            print(f"No se encontraron interacciones significativas para el usuario {user_id}.")
            return pd.DataFrame()

        cosine_sim = cosine_similarity([user_profile], self.tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        recommended_indices = [i[0] for i in sim_scores[:top_n]]
        return self.intervenciones.iloc[recommended_indices]
    

# Clase para la recomendación basada en contenido
class ContentBasedRecommender:
    def __init__(self, intervenciones, interactions_matrix):
        self.intervenciones = intervenciones
        self.interactions_matrix = interactions_matrix
        self.vectorizer = TfidfVectorizer(max_features=10000, min_df=1, max_df=0.9, ngram_range=(1,2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.intervenciones['Tags'])

    def build_user_profile(self, user_id):
        # Obtener las intervenciones que el usuario ha valorado
        user_ratings = self.interactions_matrix.loc[user_id]
        user_ratings = user_ratings[user_ratings > 0]  # Filtrar solo intervenciones valoradas

        # Si el usuario no tiene valoraciones, devolver None
        if user_ratings.empty:
            return None

        # Construir el perfil del usuario como la media ponderada de los TF-IDF de las intervenciones valoradas
        user_profile = np.zeros(self.tfidf_matrix.shape[1])

        for intervencion_id, rating in user_ratings.items():
            idx = self.intervenciones[self.intervenciones['IdIntervencion'] == intervencion_id].index[0]
            user_profile += self.tfidf_matrix[idx].toarray().flatten() * rating

        user_profile = user_profile / user_ratings.sum()  # Normalizar por la suma de las valoraciones
        return user_profile

    def recommend(self, user_id, top_n=5):
        user_profile = self.build_user_profile(user_id)
        if user_profile is None:
            print(f"No se encontraron interacciones significativas para el usuario {user_id}.")
            return None

        # Calcular la similitud entre el perfil del usuario y todas las intervenciones
        cosine_sim = cosine_similarity([user_profile], self.tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        recommended_indices = [i[0] for i in sim_scores[:top_n]]

        return self.intervenciones.iloc[recommended_indices]


# Clase para el filtrado colaborativo user-based
class UserBasedRecommender:
    def __init__(self, interactions_matrix):
        self.interactions_matrix = interactions_matrix.drop(columns=[0], errors='ignore')
        self.knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
        self.knn.fit(self.interactions_matrix)

    def recommend(self, target_user_id):
        if target_user_id not in self.interactions_matrix.index:
            print(f"El usuario con ID {target_user_id} no tiene interacciones registradas. Mostrando recomendaciones predeterminadas.")
            return self.interactions_matrix.mean(axis=0).sort_values(ascending=False).index[:5]

        user_index = self.interactions_matrix.index.get_loc(target_user_id)
        distances, indices = self.knn.kneighbors([self.interactions_matrix.iloc[user_index]], n_neighbors=6)

        similar_users = indices.flatten()[1:]

        recommended_items = self.interactions_matrix.iloc[similar_users].mean(axis=0)
        recommended_items = recommended_items.sort_values(ascending=False)[:5]

        return recommended_items.index

# Clase para el filtrado colaborativo item-based
class ItemBasedRecommender:
    def __init__(self, interactions_matrix):
        self.interactions_matrix = interactions_matrix.drop(columns=[0], errors='ignore')
        self.item_similarity = cosine_similarity(self.interactions_matrix.T)

    def recommend(self, target_user_id):
        user_ratings = self.interactions_matrix.loc[target_user_id]
        weighted_scores = self.item_similarity @ user_ratings

        recommendations = pd.Series(weighted_scores, index=self.interactions_matrix.columns)
        recommendations = recommendations.astype(float)

        recommendations = recommendations.drop(user_ratings[user_ratings > 0].index)

        return recommendations.nlargest(5).index

# Clase para la recomendación híbrida

class HybridRecommender:
    def __init__(self, users, intervenciones, interactions_matrix, usuario, weights):
        self.content_recommender = ContentBasedRecommender(intervenciones, interactions_matrix)
        self.user_based_recommender = UserBasedRecommender(interactions_matrix)
        self.item_based_recommender = ItemBasedRecommender(interactions_matrix)
        self.usuario = usuario
        self.intervenciones = intervenciones
        self.weights = weights

    def recommend(self, target_user_id):
        content_weight, user_weight, item_weight = self.weights

        content_recs = pd.Series(self.content_recommender.recommend(target_user_id)['IdIntervencion'])
        user_based_recs = pd.Series(self.user_based_recommender.recommend(target_user_id))
        item_based_recs = pd.Series(self.item_based_recommender.recommend(target_user_id))

        content_recs.index = content_recs.values
        user_based_recs.index = user_based_recs.values
        item_based_recs.index = item_based_recs.values

        combined_recs = pd.Series(dtype='float64')

        for idx in content_recs.index:
            combined_recs[idx] = combined_recs.get(idx, 0) + content_recs[idx] * content_weight
        for idx in user_based_recs.index:
            combined_recs[idx] = combined_recs.get(idx, 0) + user_based_recs[idx] * user_weight
        for idx in item_based_recs.index:
            combined_recs[idx] = combined_recs.get(idx, 0) + item_based_recs[idx] * item_weight

        combined_recs = combined_recs.sort_values(ascending=False)

        final_recommendations_df = pd.DataFrame({
            'IdIntervencion': combined_recs.index,
            'Puntuacion': combined_recs.values
        }).merge(self.intervenciones[['IdIntervencion', 'TituloApp']], on='IdIntervencion', how='left')

        return final_recommendations_df[['IdIntervencion', 'TituloApp', 'Puntuacion']]



def evaluate_model(hybrid_recommender, target_user_id):
    """
    Esta función simula la evaluación del modelo, devolviendo algunas métricas de rendimiento.
    Debes implementar las métricas adecuadas como precisión, recall, F1, diversidad y novelty.
    """
    recommendations = hybrid_recommender.recommend(target_user_id)
    # Aquí calcularás las métricas como precisión@5, recall@5, F1-score@5, diversidad, novelty, etc.
    # Por ejemplo:
    precision_at_5 = 0.4  # Este valor es un placeholder
    recall_at_5 = 0.2  # Este valor es un placeholder
    f1_score_at_5 = 0.25  # Este valor es un placeholder
    diversity = 0.95  # Este valor es un placeholder
    novelty = 0.85  # Este valor es un placeholder

    return precision_at_5, recall_at_5, f1_score_at_5, diversity, novelty



def precision_at_k(recommended_items, relevant_items, k):
    if not isinstance(recommended_items, list):
        print("Error: recommended_items no es una lista")
        return 0  # Devuelve 0 o algún valor predeterminado si no es una lista

    recommended_items_at_k = recommended_items[:k]
    relevant_and_recommended = set(recommended_items_at_k) & set(relevant_items)
    precision = len(relevant_and_recommended) / len(recommended_items_at_k)
    return precision


def recall_at_k(recommended_items, relevant_items, k):
    recommended_items_at_k = recommended_items[:k]
    relevant_and_recommended = set(recommended_items_at_k) & set(relevant_items)
    recall = len(relevant_and_recommended) / len(relevant_items)
    return recall

def f1_score_at_k(recommended_items, relevant_items, k):
    precision = precision_at_k(recommended_items, relevant_items, k)
    recall = recall_at_k(recommended_items, relevant_items, k)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def diversity(recommended_items, similarity_matrix, id_to_index_map):
    if len(recommended_items) < 2:
        return 1  # Máxima diversidad si solo se recomienda un ítem.
    
    distances = []
    for i in range(len(recommended_items)):
        for j in range(i + 1, len(recommended_items)):
            item_i = recommended_items[i]
            item_j = recommended_items[j]

            # Verifica que ambos ítems estén en el mapeo
            if item_i in id_to_index_map and item_j in id_to_index_map:
                index_i = id_to_index_map[item_i]
                index_j = id_to_index_map[item_j]
                similarity = similarity_matrix[index_i, index_j]
                distances.append(1 - similarity)
    
    return np.mean(distances) if distances else 1  # Evitar dividir por cero si distances está vacío


def novelty(recommended_items, item_popularity, k):
    novelty_score = sum(1 / np.log(1 + item_popularity[item]) for item in recommended_items[:k])
    return novelty_score / k

def adjust_for_novelty(recommendations, item_popularity, novelty_weight=0.4):
    for idx, score in recommendations.iterrows():
        item_id = score['IdIntervencion']
        popularity = item_popularity.get(item_id, 1)
        novelty_adjustment = 1 / np.log(1 + popularity)
        recommendations.at[idx, 'Puntuacion'] *= (1 + novelty_weight * novelty_adjustment)
    return recommendations.sort_values('Puntuacion', ascending=False)


def adjust_for_diversity(recommendations, similarity_matrix, id_to_index_map, diversity_weight=0.3):
    penalty = np.zeros(len(recommendations))
    for i in range(len(recommendations)):
        for j in range(i + 1, len(recommendations)):
            item_i = recommendations.iloc[i]['IdIntervencion']
            item_j = recommendations.iloc[j]['IdIntervencion']
            if item_i in id_to_index_map and item_j in id_to_index_map:
                index_i = id_to_index_map[item_i]
                index_j = id_to_index_map[item_j]
                similarity = similarity_matrix[index_i, index_j]
                penalty[j] += similarity * diversity_weight
    recommendations['Puntuacion'] -= penalty
    return recommendations.sort_values('Puntuacion', ascending=False)


def evaluate_system(recommendations, relevant_items, similarity_matrix, item_popularity, id_to_index_map, k=5):
    precision = precision_at_k(recommendations, relevant_items, k)
    recall = recall_at_k(recommendations, relevant_items, k)
    f1 = f1_score_at_k(recommendations, relevant_items, k)
    diversity_score = diversity(recommendations, similarity_matrix, id_to_index_map)
    novelty_score = novelty(recommended_items, item_popularity, k)

    print(f"Precisión@{k}: {precision}")
    print(f"Recall@{k}: {recall}")
    print(f"F1-Score@{k}: {f1}")
    print(f"Diversidad: {diversity_score}")
    print(f"Novelty: {novelty_score}")


if __name__ == "__main__":
    
    # Conectar y extraer datos
    db_manager = ConexionBaseDeDatos()
    usuario = Usuario(id_usuario='usuario_400', conexion_db=db_manager)
    users, intervenciones, tags, interacciones = db_manager.fetch_data(usuario.intervenciones_no_recomendadas)
    db_manager.desconectar()

    # Verificar si alguna de las DataFrames está vacía
    if any(df is None or df.empty for df in [users, intervenciones, tags, interacciones]):
        print("Error: No se pudo cargar todos los datos necesarios.")
        interactions_matrix, intervenciones_con_tags = None, None
    else:
        preprocessor = DataPreprocessor(users, intervenciones, tags, interacciones)
        interactions_matrix, intervenciones_con_tags = preprocessor.preprocess()

        if interactions_matrix is not None and not interactions_matrix.empty:
            
            weights=[0.2, 0.5, 0.3]
            hybrid_recommender = HybridRecommender(users, intervenciones_con_tags, interactions_matrix, usuario, weights)
            recommendations = hybrid_recommender.recommend(target_user_id=usuario.id_usuario)
            
            print("Recomendaciones híbridas con títulos:")
            print(recommendations)

            recommended_items = recommendations['IdIntervencion'].tolist()
            relevant_items = usuario.intervenciones_completadas

            print("Tipo de recommended_items:", type(recommended_items))
            print("Contenido de recommended_items:", recommended_items)

            
            item_popularity = interacciones['IdIntervencion'].value_counts().to_dict()
            recommendations = adjust_for_novelty(recommendations, item_popularity, novelty_weight=0.5)

            

            similarity_matrix = cosine_similarity(interactions_matrix.T)
            

            print("Ítems recomendados:", recommended_items)
            
            print(similarity_matrix.shape)


            print("Evaluación del sistema de recomendaciones:")

            
            # Verificar que `recommended_items` es una lista
            if isinstance(recommended_items, list):
                print("recommended_items es una lista con longitud:", len(recommended_items))
            else:
                print("Error: recommended_items no es una lista, es de tipo:", type(recommended_items))
                print("Contenido de recommended_items:", recommended_items)

            # Añadir chequeo antes de acceder a los índices
            # Generar el mapeo de IdIntervencion a índice de la matriz
            id_to_index_map = {id_: idx for idx, id_ in enumerate(interactions_matrix.columns)}
            recommendations = adjust_for_diversity(recommendations, similarity_matrix, id_to_index_map, diversity_weight=0.5)

            evaluate_system(recommended_items, relevant_items, similarity_matrix, item_popularity, id_to_index_map)

           # Definición de los pesos para GridSearch
            param_grid = {
                'content_weight': [0.2, 0.3, 0.4],
                'user_weight': [0.3, 0.4, 0.5],
                'item_weight': [0.2, 0.3, 0.4]
            }

            best_score = 0
            best_weights = None
            grid = ParameterGrid(param_grid)

            # Iterar sobre las combinaciones de pesos en el GridSearch
            for params in grid:
                content_weight = params['content_weight']
                user_weight = params['user_weight']
                item_weight = params['item_weight']
                
                # Asegurarse de que la suma de los pesos sea 1.0 (o cerca)
                if round(content_weight + user_weight + item_weight, 1) == 1.0:
                    hybrid_recommender = HybridRecommender(users, intervenciones_con_tags, interactions_matrix, usuario,
                                                        weights=(content_weight, user_weight, item_weight))
                    # Evaluar el modelo con los pesos actuales
                    precision, recall, f1_score, diversity, novelty = evaluate_model(hybrid_recommender, usuario.id_usuario)
                    
                    # Definir el criterio para el "mejor" modelo (en este caso, F1-score)
                    current_score = f1_score + diversity + novelty  # Métrica compuesta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_weights = (content_weight, user_weight, item_weight)

            print(f"Mejores pesos encontrados: {best_weights} con métrica compuesta: {best_score}")

            # Usar los mejores pesos en el sistema
            final_hybrid_recommender = HybridRecommender(users, intervenciones_con_tags, interactions_matrix, usuario,
                                                        weights=best_weights)
            recommendations = final_hybrid_recommender.recommend(usuario.id_usuario)
            print("Recomendaciones ajustadas con los mejores pesos:")
            print(recommendations)


        else:
            print("No se pudo proceder con las recomendaciones debido a la falta de datos.")

        