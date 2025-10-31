# main.py

# --- LIBRERÍAS ---
import streamlit as st
import pandas as pd
import numpy as np
import re
import googlemaps
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import time
from typing import List, Tuple, Optional
import random
from deap import base, creator, tools, algorithms
from dotenv import load_dotenv
import os

# --- LIBRERÍAS PARA POINTER NETWORK (RL) ---
import torch
import torch.nn as nn
import torch.optim as optim

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Optimizador de Rutas Múltiple", layout="wide")

# --- INTERFAZ DE USUARIO (UI) ---
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image("logotipo-oficial-programadelfin.png", width=120)
with col2:
    st.markdown("<h1 style='text-align: center;'>Optimizador de Rutas: Comparativa de 4 Métodos</h1>", unsafe_allow_html=True)
with col3:
    st.image("logotipo-oficial-verano2025.png", width=120)

st.markdown("""
Esta aplicación calcula las rutas más eficientes para la recolección de muestras, comparando cuatro enfoques distintos:
1.  **Solver Clásico (Google OR-Tools con Tiempo Real)**: Usa datos de tráfico de Google Maps para optimizar el tiempo de viaje.
2.  **IA Híbrida (Genético + Búsqueda Local)**: Combina la evolución de un algoritmo genético con una heurística de mejora local.
3.  **IA con Reinforcement Learning (PointerNet)**: Un modelo de Deep Learning entrenado para resolver problemas de ruteo.
4.  **Solver Clásico (OR-Tools con Distancia Euclidiana)**: Una versión básica que optimiza la distancia en línea recta, para comparar.
""")

# --- CARGA DE CLAVE API Y CONFIGURACIÓN DE DISPOSITIVO ---
try:
    API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]
except (KeyError, FileNotFoundError):
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if not API_KEY or API_KEY.strip() == "":
    st.error("🚨 No se encontró la API Key de Google Maps. Asegúrate de configurarla en los 'Secrets' de Streamlit o en tu archivo .env local.")
    st.stop()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.info(f"Usando dispositivo para cómputo de IA (PointerNet): **{str(device).upper()}**")

# --- CLASES Y FUNCIONES DE UTILIDAD ---
class Hospital:
    def __init__(self, etiqueta: str, nombre: str, lat: float, lon: float, capacidad: float):
        self.etiqueta = etiqueta
        self.nombre = nombre
        self.lat = lat
        self.lon = lon
        self.capacidad = capacidad

def extraer_lat_lon(wkt: str) -> Tuple[Optional[float], Optional[float]]:
    match = re.search(r'POINT\s*\(([-\d\.]+)\s+([-\d\.]+)', str(wkt))
    if match:
        lon, lat = float(match.group(1)), float(match.group(2))
        return lat, lon
    return None, None

def formato_hms_str(segundos: float) -> str:
    segundos = int(segundos)
    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segundos_restantes = segundos % 60
    if horas > 0:
        return f"{horas}h {minutos}min {segundos_restantes}s"
    elif minutos > 0:
        return f"{minutos}min {segundos_restantes}s"
    else:
        return f"{segundos_restantes}s"

def mostrar_enlaces_google_maps(ruta_info):
    coords = [(h.lat, h.lon) for h in ruta_info['hospitales']]
    MAX_PARADAS_POR_ENLACE = 10 
    
    if len(coords) > MAX_PARADAS_POR_ENLACE:
        st.markdown("**La ruta es muy larga para un solo enlace. Se ha dividido en varias partes:**")
        for i_chunk in range(0, len(coords), MAX_PARADAS_POR_ENLACE - 1):
            chunk_coords = coords[i_chunk : i_chunk + MAX_PARADAS_POR_ENLACE]
            if len(chunk_coords) < 2: continue
            chunk_labels = [h.etiqueta for h in ruta_info['hospitales'][i_chunk : i_chunk + MAX_PARADAS_POR_ENLACE]]
            url_chunk = "https://www.google.com/maps/dir/" + "/".join([f"{lat},{lon}" for lat, lon in chunk_coords])
            st.markdown(f"[Parte {i_chunk // (MAX_PARADAS_POR_ENLACE - 1) + 1}: ({' → '.join(chunk_labels)})]({url_chunk})", unsafe_allow_html=True)
    else:
        url = "https://www.google.com/maps/dir/" + "/".join([f"{lat},{lon}" for lat, lon in coords])
        st.markdown(f"[Ver esta ruta en Google Maps]({url})", unsafe_allow_html=True)

# --- CLASE 1: SOLVER CLÁSICO (OR-TOOLS CON TIEMPO REAL DE GOOGLE) ---
class OrToolsOptimizer:
    # ... (Sin cambios en esta clase)
    def __init__(self, hospitales, distance_matrix, time_matrix, max_capacidad):
        self.hospitales = hospitales
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.max_capacidad = max_capacidad
        self.demands = [int(h.capacidad) for h in hospitales]
        self.num_locations = len(hospitales)
        self.depot_index = 0

    def solve(self):
        if self.num_locations <= 1:
            return None, 0, 0
            
        num_vehicles = self.num_locations - 1 if self.num_locations > 1 else 1
        manager = pywrapcp.RoutingIndexManager(self.num_locations, num_vehicles, self.depot_index)
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.time_matrix[from_node, to_node])

        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return self.demands[from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, [self.max_capacidad] * num_vehicles, True, 'Capacidad'
        )

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(10)

        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self.process_solution(solution, manager, routing)
        else:
            return None, 0, 0

    def process_solution(self, solution, manager, routing):
        rutas_generadas = []
        distancia_total = 0
        tiempo_total = 0
        for vehicle_id in range(routing.vehicles()):
            index = routing.Start(vehicle_id)
            ruta_nodos = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                ruta_nodos.append(node_index)
                index = solution.Value(routing.NextVar(index))
            ruta_nodos.append(manager.IndexToNode(index))
            
            if len(ruta_nodos) > 2:
                carga_ruta = sum(self.demands[i] for i in ruta_nodos)
                distancia_ruta = 0
                tiempo_ruta = 0
                for i in range(len(ruta_nodos) - 1):
                    from_node, to_node = ruta_nodos[i], ruta_nodos[i+1]
                    distancia_ruta += self.distance_matrix[from_node, to_node]
                    tiempo_ruta += self.time_matrix[from_node, to_node]
                
                rutas_generadas.append({
                    "nodos": ruta_nodos,
                    "hospitales": [self.hospitales[i] for i in ruta_nodos],
                    "carga": carga_ruta,
                    "distancia": distancia_ruta,
                    "tiempo": tiempo_ruta
                })
                distancia_total += distancia_ruta
                tiempo_total += tiempo_ruta
        return rutas_generadas, distancia_total, tiempo_total

# --- CLASE 2: IA HÍBRIDA (ALGORITMO GENÉTICO + BÚSQUEDA LOCAL) - CORREGIDA ---
class GeneticAlgorithmOptimizer:
    def __init__(self, hospitales, distance_matrix, time_matrix, max_capacidad):
        self.hospitales = hospitales
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.max_capacidad = max_capacidad
        self.demands = [int(h.capacidad) for h in hospitales]
        self.num_locations = len(hospitales)
        self.depot_index = 0
        
        self.real_hospital_indices = list(range(1, self.num_locations))

        if not self.real_hospital_indices: 
            self.toolbox = None
            return

        if hasattr(creator, "FitnessMin"): del creator.FitnessMin
        if hasattr(creator, "Individual"): del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        
        num_genes = len(self.real_hospital_indices)
        self.toolbox.register("indices", random.sample, range(num_genes), num_genes)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self.eval_cvrp_hybrid)
        self.toolbox.register("mate", tools.cxOrdered)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    # ARREGLO 1: Algoritmo 2-Opt mucho más eficiente y con límite de iteraciones
    def local_search_2_opt(self, route_nodes, max_iterations=1000):
        if len(route_nodes) <= 3:
            return route_nodes
        
        best_route = route_nodes
        iteration = 0
        improved = True
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    # Calcula solo el cambio en el tiempo (delta), no la suma total
                    current_edges = (self.time_matrix[best_route[i-1], best_route[i]] + 
                                     self.time_matrix[best_route[j], best_route[j+1]])
                    new_edges = (self.time_matrix[best_route[i-1], best_route[j]] + 
                                 self.time_matrix[best_route[i], best_route[j+1]])

                    if new_edges < current_edges:
                        # Aplica el intercambio (inversión)
                        best_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                        improved = True
                        # Salta al siguiente ciclo while para reiniciar la búsqueda desde el principio
                        break 
                if improved:
                    break
        return best_route

    def build_routes_from_individual(self, individual):
        routes = []
        current_route = [self.depot_index]
        current_load = 0
        
        for idx_in_individual in individual:
            hospital_idx = self.real_hospital_indices[idx_in_individual]
            
            if current_load + self.demands[hospital_idx] <= self.max_capacidad:
                current_route.append(hospital_idx)
                current_load += self.demands[hospital_idx]
            else:
                current_route.append(self.depot_index)
                routes.append(current_route)
                current_route = [self.depot_index, hospital_idx]
                current_load = self.demands[hospital_idx]
        
        current_route.append(self.depot_index)
        routes.append(current_route)
        return routes

    def eval_cvrp_hybrid(self, individual):
        initial_routes = self.build_routes_from_individual(individual)
        total_time = 0
        for route in initial_routes:
            # Llama a la versión optimizada de 2-Opt
            optimized_route = self.local_search_2_opt(route)
            for i in range(len(optimized_route) - 1):
                total_time += self.time_matrix[optimized_route[i], optimized_route[i+1]]
        return (total_time,)

    def solve(self, ngen=100, pop_size=50, cxpb=0.7, mutpb=0.2):
        if not self.real_hospital_indices or not self.toolbox: 
            return [], 0, 0
            
        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        algorithms.eaSimple(pop, self.toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, halloffame=hof, verbose=False)
        return self.process_solution(hof[0])

    def process_solution(self, individual):
        initial_routes = self.build_routes_from_individual(individual)
        rutas_generadas, distancia_total, tiempo_total = [], 0, 0
        
        for route in initial_routes:
            optimized_route_nodes = self.local_search_2_opt(route)
            distancia_ruta = sum(self.distance_matrix[optimized_route_nodes[i], optimized_route_nodes[i+1]] for i in range(len(optimized_route_nodes)-1))
            tiempo_ruta = sum(self.time_matrix[optimized_route_nodes[i], optimized_route_nodes[i+1]] for i in range(len(optimized_route_nodes)-1))
            carga_ruta = sum(self.demands[node] for node in optimized_route_nodes)
            
            rutas_generadas.append({
                "nodos": optimized_route_nodes,
                "hospitales": [self.hospitales[i] for i in optimized_route_nodes],
                "carga": carga_ruta,
                "distancia": distancia_ruta,
                "tiempo": tiempo_ruta
            })
            distancia_total += distancia_ruta
            tiempo_total += tiempo_ruta
            
        return rutas_generadas, distancia_total, tiempo_total

# --- CLASE 3: POINTER NETWORK (REINFORCEMENT LEARNING) ---
# ... (Sin cambios en esta clase)
class PointerNetRL(nn.Module):
    def __init__(self, input_dim=3, embedding_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.pointer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)
        enc_outputs, (h, c) = self.encoder(emb)
        dec_input = enc_outputs[:, 0:1, :]
        outs = []
        for _ in range(x.size(1)):
            dec_output, (h, c) = self.decoder(dec_input, (h, c))
            attn_logits = self.pointer(dec_output.squeeze(1)).unsqueeze(1)
            outs.append(attn_logits.squeeze(1))
            dec_input = dec_output 
        outs = torch.cat(outs, 1)
        return outs

@st.cache_resource(show_spinner="Entrenando modelo RL (esto se almacena en caché)...")
def train_pointer_net_model(epochs, lr):
    model = PointerNetRL(input_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def generate_vrp_instance(n_nodes=10, demand_low=1, demand_high=10):
        coords = np.random.rand(n_nodes + 1, 2)
        demands = np.random.randint(demand_low, demand_high+1, n_nodes + 1)
        demands[0] = 0
        return coords, demands

    def reinforce_loss(model, coords, demands, capacity):
        n = len(coords)
        inputs = torch.tensor(np.concatenate([coords, demands.reshape(-1,1)/capacity], axis=1), dtype=torch.float).unsqueeze(0).to(device)
        logits = model(inputs).squeeze(0)
        
        logprobs, rewards = [], []
        
        for _ in range(5): 
            route, visited, curr_cap, cost, curr_node = [], {0}, capacity, 0, 0
            
            for _ in range(n - 1):
                available_indices = [i for i in range(n) if i not in visited and demands[i] <= curr_cap]
                if not available_indices:
                    cost += np.linalg.norm(coords[curr_node] - coords[0])
                    curr_node, curr_cap = 0, capacity
                    available_indices = [i for i in range(n) if i not in visited]

                if not available_indices: break
                
                available_logits = logits[available_indices]
                probs = torch.softmax(available_logits, dim=0)
                m = torch.distributions.Categorical(probs)
                action = m.sample()
                next_node = available_indices[action.item()]

                logprobs.append(m.log_prob(action))
                cost += np.linalg.norm(coords[curr_node] - coords[next_node])
                curr_cap -= demands[next_node]
                visited.add(next_node)
                curr_node = next_node
                route.append(next_node)

            cost += np.linalg.norm(coords[curr_node] - coords[0])
            rewards.append(-cost)

        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        logprobs = torch.stack(logprobs)
        loss = - (rewards.mean() * logprobs.mean())
        if torch.isnan(loss) or torch.isinf(loss): return None
        return loss

    for epoch in range(epochs):
        coords_sim, demands_sim = generate_vrp_instance(n_nodes=10)
        loss = reinforce_loss(model, coords_sim, demands_sim, 40)
        if loss is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

class PointerNetRLOptimizer:
    def __init__(self, hospitales, distance_matrix, time_matrix, max_capacidad, model):
        self.hospitales = hospitales
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.max_capacidad = max_capacidad
        self.model = model
        self.demands = np.array([h.capacidad for h in hospitales])
        self.coords = np.array([(h.lon, h.lat) for h in hospitales])
        self.num_locations = len(hospitales)

    def solve(self):
        if self.num_locations <= 1:
            return [], 0, 0
        self.model.eval()
        inputs = torch.tensor(np.concatenate([self.coords, self.demands.reshape(-1, 1) / self.max_capacidad], axis=1), dtype=torch.float).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.model(inputs).squeeze(0).cpu().numpy()
        
        routes = []
        unvisited = list(range(1, self.num_locations))
        
        while unvisited:
            current_route_nodes = [0]
            current_load = 0
            sorted_unvisited = sorted(unvisited, key=lambda i: logits[i], reverse=True)
            
            for node_idx in sorted_unvisited:
                if node_idx in unvisited and current_load + self.demands[node_idx] <= self.max_capacidad:
                    current_route_nodes.append(node_idx)
                    current_load += self.demands[node_idx]
                    unvisited.remove(node_idx)

            current_route_nodes.append(0)
            routes.append(current_route_nodes)
        
        return self.process_solution(routes)

    def process_solution(self, generated_routes):
        final_routes, distancia_total, tiempo_total = [], 0, 0
        for route_nodes in generated_routes:
            if len(route_nodes) > 2:
                distancia_ruta = sum(self.distance_matrix[route_nodes[i], route_nodes[i+1]] for i in range(len(route_nodes)-1))
                tiempo_ruta = sum(self.time_matrix[route_nodes[i], route_nodes[i+1]] for i in range(len(route_nodes)-1))
                carga_ruta = sum(self.demands[node] for node in route_nodes)
                
                final_routes.append({
                    "nodos": route_nodes,
                    "hospitales": [self.hospitales[i] for i in route_nodes],
                    "carga": carga_ruta,
                    "distancia": distancia_ruta,
                    "tiempo": tiempo_ruta
                })
                distancia_total += distancia_ruta
                tiempo_total += tiempo_ruta

        return final_routes, distancia_total, tiempo_total

# --- CLASE 4: OR-TOOLS CON DISTANCIA EUCLIDIANA ---
# ... (Sin cambios en esta clase)
class OrToolsEuclideanOptimizer:
    def __init__(self, hospitales, max_capacidad, real_dist_matrix, real_time_matrix):
        self.hospitales = hospitales
        self.coords = np.array([(h.lon, h.lat) for h in hospitales])
        self.dist_matrix_euc = np.linalg.norm(self.coords[:, None, :] - self.coords[None, :, :], axis=-1)
        self.real_dist_matrix = real_dist_matrix 
        self.real_time_matrix = real_time_matrix
        self.max_capacidad = max_capacidad
        self.demands = [int(h.capacidad) for h in hospitales]
        self.num_locations = len(hospitales)
        self.depot_index = 0

    def solve(self):
        if self.num_locations <= 1:
            return None, 0, 0

        num_vehicles = self.num_locations - 1 if self.num_locations > 1 else 1
        manager = pywrapcp.RoutingIndexManager(self.num_locations, num_vehicles, self.depot_index)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.dist_matrix_euc[from_node, to_node] * 100000)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return self.demands[from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, [self.max_capacidad] * num_vehicles, True, 'Capacidad'
        )
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self.process_solution(solution, manager, routing)
        else:
            return None, 0, 0

    def process_solution(self, solution, manager, routing):
        rutas_generadas, distancia_total_real, tiempo_total_real = [], 0, 0
        for vehicle_id in range(routing.vehicles()):
            index = routing.Start(vehicle_id)
            ruta_nodos = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                ruta_nodos.append(node_index)
                index = solution.Value(routing.NextVar(index))
            ruta_nodos.append(manager.IndexToNode(index))
            
            if len(ruta_nodos) > 2:
                dist_real = sum(self.real_dist_matrix[ruta_nodos[i], ruta_nodos[i+1]] for i in range(len(ruta_nodos)-1))
                tiempo_real = sum(self.real_time_matrix[ruta_nodos[i], ruta_nodos[i+1]] for i in range(len(ruta_nodos)-1))

                rutas_generadas.append({
                    "nodos": ruta_nodos,
                    "hospitales": [self.hospitales[i] for i in ruta_nodos],
                    "carga": sum(self.demands[i] for i in ruta_nodos),
                    "distancia": dist_real,
                    "tiempo": tiempo_real
                })
                distancia_total_real += dist_real
                tiempo_total_real += tiempo_real
        return rutas_generadas, distancia_total_real, tiempo_total_real

# --- FUNCIÓN MEJORADA PARA OBTENER MATRICES DE GOOGLE MAPS ---
# ... (Sin cambios en esta clase)
@st.cache_data
def get_matrices(hospitales_data_tuple: Tuple[Tuple, ...], api_key: str):
    locations = [(h[2], h[3]) for h in hospitales_data_tuple]
    hospital_names = [h[1] for h in hospitales_data_tuple]
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    time_matrix = np.zeros((n, n))
    gmaps = googlemaps.Client(key=api_key)
    CHUNK_SIZE = 25

    for i in range(n):
        origin = [locations[i]]
        for j_start in range(0, n, CHUNK_SIZE):
            j_end = min(j_start + CHUNK_SIZE, n)
            destinations_chunk = locations[j_start:j_end]
            if not destinations_chunk: continue
            try:
                response = gmaps.distance_matrix(origin, destinations_chunk, mode="driving", departure_time="now")
                if response['status'] == 'OK' and response['rows']:
                    row_elements = response['rows'][0]['elements']
                    for k, element in enumerate(row_elements):
                        j = j_start + k
                        if element['status'] == 'OK':
                            distance_matrix[i, j] = element.get('distance', {}).get('value', np.nan)
                            duration_info = element.get('duration_in_traffic') or element.get('duration')
                            if duration_info and 'value' in duration_info:
                                time_matrix[i, j] = duration_info['value']
                            else:
                                time_matrix[i, j] = np.nan
                        else:
                            distance_matrix[i, j], time_matrix[i, j] = np.nan, np.nan
                            st.warning(f"Google no pudo encontrar ruta entre '{hospital_names[i]}' y '{hospital_names[j]}'. Estado: {element['status']}")
            except Exception as e:
                st.error(f"Error en API de Google para el origen '{hospital_names[i]}' y el destino en el índice {j_start}: {e}")
                distance_matrix[i, j_start:j_end].fill(np.nan)
                time_matrix[i, j_start:j_end].fill(np.nan)
    return distance_matrix, time_matrix

# --- LÓGICA PRINCIPAL DE LA APP ---
with st.sidebar:
    # ... (Lógica de la sidebar sin cambios, ya era robusta)
    st.header("📄 Sube tu archivo")
    st.markdown("Asegúrate de que tu archivo Excel tenga estas columnas: `ID`, `Nombre`, `WKT`, `Peso en Kg`.")
    uploaded_file = st.file_uploader("Sube el archivo de localizaciones (.xlsx)", type=["xlsx"])
    
    calcular = None
    seleccionados = []
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            columnas_requeridas = ['ID', 'Nombre', 'WKT', 'Peso en Kg']
            if not all(col in df.columns for col in columnas_requeridas):
                st.error(f"El archivo subido no es válido. Faltan las columnas: {', '.join([c for c in columnas_requeridas if c not in df.columns])}.")
                st.stop()
            
            df['Peso en Kg'] = pd.to_numeric(df['Peso en Kg'], errors='coerce').fillna(0)
            todos_los_hospitales = [Hospital(str(r['ID']).strip(), str(r['Nombre']).strip(), *extraer_lat_lon(r['WKT']), float(r['Peso en Kg'])) for _, r in df.iterrows() if extraer_lat_lon(r['WKT'])[0] is not None]

            if not todos_los_hospitales:
                st.warning("El archivo no contiene ninguna localización válida. Revisa los datos.")
                st.stop()

            st.header("✅ Configuración de la Simulación")
            
            depot_idx = st.selectbox("1. Elige el Centro de Acopio (Punto de Partida):", options=range(len(todos_los_hospitales)), format_func=lambda i: f"{todos_los_hospitales[i].etiqueta} - {todos_los_hospitales[i].nombre}")
            
            depot_hospital = todos_los_hospitales.pop(depot_idx)
            depot_hospital.capacidad = 0
            
            hospitales_para_visitar = todos_los_hospitales
            
            if hospitales_para_visitar:
                if st.checkbox("Seleccionar todos los puntos a visitar", value=True):
                    seleccionados_idx_default = list(range(len(hospitales_para_visitar)))
                else:
                    seleccionados_idx_default = []

                seleccionados_idx_visitar = st.multiselect(
                    "2. Elige los puntos a visitar:", 
                    options=list(range(len(hospitales_para_visitar))), 
                    format_func=lambda i: f"{hospitales_para_visitar[i].etiqueta} - {hospitales_para_visitar[i].nombre}",
                    default=seleccionados_idx_default
                )
                
                seleccionados = [depot_hospital] + [hospitales_para_visitar[i] for i in seleccionados_idx_visitar]
            else:
                st.warning("No hay puntos disponibles para visitar (solo se encontró el depósito).")
                seleccionados = [depot_hospital]

            MAX_CAPACIDAD_VEHICULO = st.number_input("3. Capacidad máxima del vehículo (Kg)", min_value=1, value=1500, step=10)
            
            with st.expander("Parámetros Avanzados de IA"):
                st.subheader("IA Híbrida (Genético)")
                NGEN = st.slider("Número de Generaciones", 50, 500, 100, step=50, key="ngen") # Valor por defecto reducido
                POP_SIZE = st.slider("Tamaño de la Población", 20, 200, 50, step=10, key="pop") # Valor por defecto reducido
                
                st.subheader("IA (Reinforcement Learning)")
                RL_EPOCHS = st.slider("Épocas de Entrenamiento RL", 500, 2500, 1000, step=100, key="rl_epochs")
                RL_LR = st.number_input("Tasa de Aprendizaje RL", 0.0001, 0.01, 0.001, format="%.4f", key="rl_lr")

            calcular = st.button("🚀 Optimizar y Comparar", use_container_width=True, type="primary")
        
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")
            st.stop()

if not uploaded_file:
    st.info("Por favor, sube tu archivo Excel para comenzar.")
    st.stop()
if not calcular:
    st.info("Configura la simulación en la barra lateral y haz clic en 'Optimizar'.")
    st.stop()
if len(seleccionados) <= 1:
    st.warning("Debes seleccionar al menos un punto para visitar (además del centro de acopio).")
    st.stop()

# --- EJECUCIÓN DE MODELOS (REFACTORIZADO) ---
st.subheader("Ubicación de Puntos Seleccionados")
df_mapa = pd.DataFrame([(h.nombre, h.lat, h.lon) for h in seleccionados], columns=['name', 'lat', 'lon'])
st.map(df_mapa)
st.markdown("---")

resultados_finales = {}

with st.spinner("Obteniendo datos de distancia y tráfico de Google Maps... Esto puede tardar si hay muchos puntos."):
    hospital_data_for_cache = tuple((h.etiqueta, h.nombre, h.lat, h.lon, h.capacidad) for h in seleccionados)
    dist_matrix, time_matrix = get_matrices(hospital_data_for_cache, API_KEY)

if np.isnan(dist_matrix).any():
    st.error("No se pudieron obtener todos los datos de la API de Google. Verifica las advertencias y no se puede continuar.")
    st.stop()

# ARREGLO 3: Bucle de ejecución más limpio y con barra de progreso
st.header("Ejecutando Modelos...")
progress_bar = st.progress(0)
status_text = st.empty()

modelos_a_ejecutar = {
    "Clásico (OR-Tools Tiempo Real)": OrToolsOptimizer(seleccionados, dist_matrix, time_matrix, MAX_CAPACIDAD_VEHICULO),
    "IA Híbrida (Genético)": GeneticAlgorithmOptimizer(seleccionados, dist_matrix, time_matrix, MAX_CAPACIDAD_VEHICULO),
    "IA (PointerNet RL)": "placeholder_rl",
    "Clásico (OR-Tools Euclidiano)": OrToolsEuclideanOptimizer(seleccionados, MAX_CAPACIDAD_VEHICULO, dist_matrix, time_matrix)
}
iconos = {"Clásico (OR-Tools Tiempo Real)": "🚚", "IA Híbrida (Genético)": "🧬", "IA (PointerNet RL)": "🤖", "Clásico (OR-Tools Euclidiano)": "📏"}
num_modelos = len(modelos_a_ejecutar)

for i, (nombre, optimizador) in enumerate(modelos_a_ejecutar.items()):
    status_text.text(f"Ejecutando modelo {i+1}/{num_modelos}: {nombre}")
    
    t_inicio = time.time()
    rutas, dist, tiempo = None, 0, 0
    
    try:
        if nombre == "IA (PointerNet RL)":
            pointer_model = train_pointer_net_model(epochs=RL_EPOCHS, lr=RL_LR)
            optimizador_rl = PointerNetRLOptimizer(seleccionados, dist_matrix, time_matrix, MAX_CAPACIDAD_VEHICULO, pointer_model)
            rutas, dist, tiempo = optimizador_rl.solve()
        elif nombre == "IA Híbrida (Genético)":
            rutas, dist, tiempo = optimizador.solve(ngen=NGEN, pop_size=POP_SIZE)
        else:
            rutas, dist, tiempo = optimizador.solve()
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al ejecutar el modelo '{nombre}': {e}")
        rutas = None # Asegurarse de que no se procesen resultados parciales

    t_fin = time.time()
    
    if rutas:
        resultados_finales[nombre] = {
            "Tiempo Total": formato_hms_str(tiempo), 
            "Distancia Total (km)": f"{dist/1000:.2f}", 
            "Tiempo de Cómputo (s)": f"{t_fin - t_inicio:.4f}", 
            "Nº de Rutas": len(rutas),
            "rutas_obj": rutas # Guardar las rutas para mostrarlas después
        }
    else:
        st.warning(f"El modelo '{nombre}' no pudo encontrar una solución o falló.")
        
    progress_bar.progress((i + 1) / num_modelos)

status_text.text("¡Todos los modelos han sido ejecutados!")
st.markdown("---")

# Bucle de visualización separado
st.header("Resultados Detallados por Modelo")
for nombre, resultado in resultados_finales.items():
    st.subheader(f"{iconos[nombre]} {nombre}")
    for j, ruta_info in enumerate(resultado["rutas_obj"]):
        expander_title = (
            f"Ruta #{j+1} | "
            f"Carga: {ruta_info['carga']} Kg | "
            f"Tiempo: {formato_hms_str(ruta_info['tiempo'])} | "
            f"Distancia: {ruta_info['distancia']/1000:.2f} km"
        )
        with st.expander(expander_title):
            st.info(" → ".join([h.etiqueta for h in ruta_info['hospitales']]))
            mostrar_enlaces_google_maps(ruta_info)
    st.markdown("---")

# Tabla Comparativa Final
if len(resultados_finales) > 1:
    st.header("📊 Comparativa Final de Modelos")
    # Eliminar el objeto de rutas antes de crear el DataFrame
    df_comp_data = {k: {k2: v2 for k2, v2 in v.items() if k2 != 'rutas_obj'} for k, v in resultados_finales.items()}
    df_comp = pd.DataFrame(df_comp_data).T
    st.table(df_comp)