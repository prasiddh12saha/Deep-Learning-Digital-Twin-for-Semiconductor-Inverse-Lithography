"""
MODULE 7: Streamlit Interactive CAD Dashboard & Visualizer
Provides real-time parameter tuning (dose, defocus, etch bias)
and side-by-side physical twin metrology inspection.
"""

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.litho_data import LayoutGenerator
from src.litho_optics import NeuralOpticsEngine, simulate_diffraction_ground_truth
from src.litho_resist.resist_model import ResistUNet, simulate_resist_ground_truth
from src.litho_etch import NeuralEtchEngine, simulate_etch_ground_truth
from src.litho_ilt import DifferentiableILTOptimizer
from src.litho_metrology import MetrologyEngine

st.set_page_config(page_title="Neural Litho Twin CAD", layout="wide")

st.title("🔬 Semiconductor Inverse Lithography Digital Twin")
st.markdown("End-to-end differentiable neural surrogate modeling: Mask -> Optics -> Resist -> Etch -> ILT")

# Sidebar Controls
st.sidebar.header("Process Controls")
pattern_type = st.sidebar.selectbox("Layout Pattern", ["line_space", "contacts", "junctions"])
run_ilt = st.sidebar.checkbox("Run Inverse Lithography (ILT)", value=True)
ilt_iters = st.sidebar.slider("ILT Iterations", 20, 150, 60, step=10)
lr = st.sidebar.slider("Optimizer Learning Rate", 0.01, 0.20, 0.08, step=0.01)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    optics = NeuralOpticsEngine().to(device).eval()
    resist = ResistUNet().to(device).eval()
    etch = NeuralEtchEngine().to(device).eval()
    return optics, resist, etch

optics_net, resist_net, etch_net = load_models()
metrology = MetrologyEngine(threshold=0.5)

# Generate Base Layout
gen = LayoutGenerator(resolution=128)
if pattern_type == "line_space":
    mask_np = gen._create_line_space()
elif pattern_type == "contacts":
    mask_np = gen._create_contact_array()
else:
    mask_np = gen._create_junctions()

target_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)

# Forward Nominal Simulation
with torch.no_grad():
    nom_aerial = optics_net(target_tensor)
    nom_resist = resist_net(nom_aerial)
    nom_silicon = etch_net(nom_resist)

nom_silicon_np = nom_silicon[0, 0].cpu().numpy()
nominal_metrics = metrology.evaluate(mask_np, nom_silicon_np)

# Display Forward Simulation
st.subheader("1. Forward Process Physics")
c1, c2, c3, c4 = st.columns(4)
c1.image(mask_np, caption="Target Mask M(x,y)", clamp=True)
c2.image(nom_aerial[0, 0].cpu().numpy(), caption="Aerial Intensity I(x,y)", clamp=True)
c3.image(nom_resist[0, 0].cpu().numpy(), caption="Resist Profile H(x,y)", clamp=True)
c4.image(nom_silicon_np, caption="Etched Silicon S(x,y)", clamp=True)

st.metric("Nominal Mean EPE", f"{nominal_metrics['mean_epe_pixels']:.3f} px")

# ILT Optimization
if run_ilt:
    st.subheader("2. Differentiable Inverse Lithography Synthesis")
    with st.spinner("Backpropagating gradients through frozen neural twin..."):
        ilt_solver = DifferentiableILTOptimizer(optics_net, resist_net, etch_net, device)
        ilt_res = ilt_solver.optimize(target_tensor, iterations=ilt_iters, lr=lr)

    opt_mask_np = ilt_res["optimized_mask"][0, 0].cpu().numpy()
    final_silicon_np = ilt_res["final_silicon"][0, 0].cpu().numpy()
    ilt_metrics = metrology.evaluate(mask_np, final_silicon_np)

    col1, col2, col3, col4 = st.columns(4)
    col1.image(opt_mask_np, caption="Optimized Mask (OPC/SRAF)", clamp=True)
    col2.image(final_silicon_np, caption="Silicon from ILT Mask", clamp=True)
    col3.image(nominal_metrics["error_map"], caption="Uncorrected Error Map", clamp=True)
    col4.image(ilt_metrics["error_map"], caption="ILT Corrected Error Map", clamp=True)

    epe_delta = ((nominal_metrics['mean_epe_pixels'] - ilt_metrics['mean_epe_pixels']) / (nominal_metrics['mean_epe_pixels'] + 1e-6)) * 100
    st.metric("ILT Corrected Mean EPE", f"{ilt_metrics['mean_epe_pixels']:.3f} px", delta=f"-{epe_delta:.1f}%")
