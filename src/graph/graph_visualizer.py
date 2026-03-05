"""
Graph Visualizer - Week 9 Day 4
Visualize knowledge graph using matplotlib and networkx.
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, List, Set
from pathlib import Path


class GraphVisualizer:
    """Visualize knowledge graphs."""
    
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize visualizer.
        
        Args:
            graph: NetworkX DiGraph to visualize
        """
        self.graph = graph
    
    def visualize(
        self,
        output_path: Optional[str] = None,
        highlight_nodes: Optional[List[str]] = None,
        highlight_path: Optional[List[str]] = None,
        figsize: tuple = (15, 10),
        node_size: int = 3000,
        font_size: int = 10
    ):
        """
        Visualize full graph.
        
        Args:
            output_path: Path to save image (if None, show instead)
            highlight_nodes: Nodes to highlight in different color
            highlight_path: Path to highlight
            figsize: Figure size
            node_size: Size of nodes
            font_size: Font size for labels
        """
        if self.graph.number_of_nodes() == 0:
            print("⚠️  Empty graph, nothing to visualize")
            return
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        
        # Prepare node colors
        node_colors = []
        for node in self.graph.nodes():
            if highlight_nodes and node in highlight_nodes:
                node_colors.append('#FF6B6B')  # Red for highlighted
            elif highlight_path and node in highlight_path:
                node_colors.append('#4ECDC4')  # Teal for path
            else:
                node_colors.append('#95E1D3')  # Light green default
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=node_size,
            alpha=0.9
        )
        
        # Draw edges
        if highlight_path:
            # Highlight path edges
            path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
            other_edges = [e for e in self.graph.edges() if e not in path_edges]
            
            # Draw path edges (thick, colored)
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=path_edges,
                edge_color='#FF6B6B',
                width=3,
                alpha=0.8,
                arrows=True,
                arrowsize=20
            )
            
            # Draw other edges (thin, gray)
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=other_edges,
                edge_color='gray',
                width=1,
                alpha=0.3,
                arrows=True,
                arrowsize=15
            )
        else:
            # Draw all edges normally
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edge_color='gray',
                width=1.5,
                alpha=0.5,
                arrows=True,
                arrowsize=20
            )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph,
            pos,
            font_size=font_size,
            font_weight='bold'
        )
        
        # Draw edge labels (relationships)
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels,
            font_size=8
        )
        
        plt.title("Knowledge Graph", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Graph saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_subgraph(
        self,
        center_nodes: List[str],
        k_hop: int = 1,
        output_path: Optional[str] = None
    ):
        """
        Visualize subgraph around specific nodes.
        
        Args:
            center_nodes: Center nodes for subgraph
            k_hop: Number of hops to include
            output_path: Path to save image
        """
        # Get nodes within k hops
        nodes = set(center_nodes)
        
        for _ in range(k_hop):
            new_nodes = set()
            for node in nodes:
                if node in self.graph:
                    new_nodes.update(self.graph.successors(node))
                    new_nodes.update(self.graph.predecessors(node))
            nodes.update(new_nodes)
        
        # Extract subgraph
        subgraph = self.graph.subgraph(nodes).copy()
        
        print(f"\n🔍 Subgraph around {center_nodes} ({k_hop}-hop):")
        print(f"   Nodes: {subgraph.number_of_nodes()}")
        print(f"   Edges: {subgraph.number_of_edges()}")
        
        # Create temporary visualizer for subgraph
        sub_viz = GraphVisualizer(subgraph)
        sub_viz.visualize(
            output_path=output_path,
            highlight_nodes=center_nodes,
            figsize=(12, 8)
        )
    
    def visualize_path(
        self,
        source: str,
        target: str,
        output_path: Optional[str] = None
    ):
        """
        Visualize path between two nodes.
        
        Args:
            source: Source node
            target: Target node
            output_path: Path to save image
        """
        if source not in self.graph or target not in self.graph:
            print(f"⚠️  Node not found in graph")
            return
        
        try:
            # Find shortest path
            path = nx.shortest_path(self.graph, source, target)
            
            print(f"\n🛤️  Path from '{source}' to '{target}':")
            print(f"   {' → '.join(path)}")
            
            # Get subgraph with path context
            nodes = set(path)
            for node in path:
                nodes.update(self.graph.successors(node))
                nodes.update(self.graph.predecessors(node))
            
            subgraph = self.graph.subgraph(nodes).copy()
            
            # Visualize
            sub_viz = GraphVisualizer(subgraph)
            sub_viz.visualize(
                output_path=output_path,
                highlight_path=path,
                figsize=(14, 10)
            )
            
        except nx.NetworkXNoPath:
            print(f"⚠️  No path found between '{source}' and '{target}'")
    
    def generate_stats_report(self) -> str:
        """Generate text statistics report."""
        
        report = []
        report.append("\n" + "="*60)
        report.append("📊 KNOWLEDGE GRAPH STATISTICS")
        report.append("="*60)
        
        # Basic stats
        report.append(f"\n🔢 Basic Metrics:")
        report.append(f"   Total Nodes: {self.graph.number_of_nodes()}")
        report.append(f"   Total Edges: {self.graph.number_of_edges()}")
        report.append(f"   Graph Density: {nx.density(self.graph):.4f}")
        
        if self.graph.number_of_nodes() > 0:
            # Connectivity
            weakly = nx.number_weakly_connected_components(self.graph)
            strongly = nx.number_strongly_connected_components(self.graph)
            report.append(f"\n🔗 Connectivity:")
            report.append(f"   Weakly Connected Components: {weakly}")
            report.append(f"   Strongly Connected Components: {strongly}")
            
            # Top nodes
            report.append(f"\n⭐ Top 10 Nodes by Degree:")
            degree_dict = dict(self.graph.degree())
            top_nodes = sorted(
                degree_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            for i, (node, degree) in enumerate(top_nodes, 1):
                report.append(f"   {i:2}. {node:20} (degree: {degree})")
            
            # Relationship types
            relations = {}
            for u, v, data in self.graph.edges(data=True):
                rel = data.get('relation', 'unknown')
                relations[rel] = relations.get(rel, 0) + 1
            
            report.append(f"\n🔀 Relationship Types:")
            for rel, count in sorted(relations.items(), key=lambda x: x[1], reverse=True):
                report.append(f"   {rel:20} : {count:3}")
        
        report.append("\n" + "="*60 + "\n")
        
        return "\n".join(report)