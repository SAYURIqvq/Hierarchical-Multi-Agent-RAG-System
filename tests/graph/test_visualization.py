"""Test Graph Visualization - Week 9 Day 4"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.graph_builder import KnowledgeGraph
from src.graph.graph_visualizer import GraphVisualizer


def test_visualization():
    """Test graph visualization."""
    
    print("\n🧪 Testing Graph Visualization\n")
    
    # Load test graph
    kg = KnowledgeGraph()
    kg.load('data/graphs/test_graph.pkl')
    
    print(f"📂 Loaded graph: {kg}")
    
    # Create visualizer
    viz = GraphVisualizer(kg.graph)
    
    # Test 1: Full graph
    print("\n1️⃣ Visualizing full graph...")
    viz.visualize(output_path='data/graphs/full_graph.png')
    
    # Test 2: Subgraph around TensorFlow
    print("\n2️⃣ Visualizing subgraph around 'tensorflow'...")
    viz.visualize_subgraph(
        center_nodes=['tensorflow'],
        k_hop=1,
        output_path='data/graphs/subgraph_tensorflow.png'
    )
    
    # Test 3: Path visualization
    print("\n3️⃣ Visualizing path from 'google' to 'machine learning'...")
    viz.visualize_path(
        source='google',
        target='machine learning',
        output_path='data/graphs/path_google_ml.png'
    )
    
    # Test 4: Statistics report
    print(viz.generate_stats_report())
    
    print("✅ All visualizations created!\n")


if __name__ == "__main__":
    test_visualization()