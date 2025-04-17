import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar, k as k_B, atomic_mass, c

# Physical constants for Rubidium-87
RB87_MASS = 86.909180527 * atomic_mass  # kg
WAVELENGTH_D2 = 780.241e-9  # m (D2 transition wavelength)
WAVENUMBER_D2 = 2 * np.pi / WAVELENGTH_D2  # m^-1
GAMMA_D2 = 2 * np.pi * 6.065e6  # s^-1 (natural linewidth of D2 transition)
RECOIL_ENERGY = (hbar * WAVENUMBER_D2)**2 / (2 * RB87_MASS)  # J
RECOIL_TEMPERATURE = RECOIL_ENERGY / k_B  # K
DOPPLER_TEMPERATURE = hbar * GAMMA_D2 / (2 * k_B)  # K

class WorldModel:
    """A data structure to represent the relationships between physical quantities and concepts."""
    
    def __init__(self):
        """Initialize the world model with a graph structure."""
        self.graph = nx.DiGraph()
        self.variables = {}
        self.equations = {}
        self.concepts = {}
    
    def add_variable(self, name, value=None, unit=None, description=None):
        """Add a physical variable to the world model."""
        self.variables[name] = {
            'value': value,
            'unit': unit,
            'description': description
        }
        self.graph.add_node(name, type='variable')
        return self
    
    def add_equation(self, name, equation, description=None, variables=None):
        """Add an equation to the world model."""
        self.equations[name] = {
            'equation': equation,
            'description': description
        }
        self.graph.add_node(name, type='equation')
        
        # Connect equation to variables
        if variables:
            for var in variables:
                if var in self.variables:
                    self.graph.add_edge(name, var)
        return self
    
    def add_concept(self, name, description=None, related_to=None):
        """Add a concept to the world model."""
        self.concepts[name] = {
            'description': description
        }
        self.graph.add_node(name, type='concept')
        
        # Connect concept to related entities
        if related_to:
            for entity in related_to:
                if entity in self.variables or entity in self.equations or entity in self.concepts:
                    self.graph.add_edge(name, entity)
        return self
    
    def connect(self, source, target):
        """Connect two entities in the world model."""
        if source in self.variables or source in self.equations or source in self.concepts:
            if target in self.variables or target in self.equations or target in self.concepts:
                self.graph.add_edge(source, target)
        return self
    
    def visualize(self, figsize=(12, 10)):
        """Visualize the world model as a graph."""
        plt.figure(figsize=figsize)
        
        # Define node colors based on type
        node_colors = []
        for node in self.graph.nodes():
            if node in self.variables:
                node_colors.append('skyblue')
            elif node in self.equations:
                node_colors.append('lightgreen')
            else:  # concept
                node_colors.append('salmon')
        
        # Draw the graph
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500, alpha=0.8)
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(self.graph, pos, font_size=10)
        
        # Add a legend
        plt.plot([], [], 'o', color='skyblue', label='Variable')
        plt.plot([], [], 'o', color='lightgreen', label='Equation')
        plt.plot([], [], 'o', color='salmon', label='Concept')
        plt.legend()
        
        plt.axis('off')
        plt.title('Sisyphus Cooling World Model')
        plt.tight_layout()
        plt.show()
    
    def get_related(self, entity_name):
        """Get entities related to the given entity."""
        if entity_name not in self.graph:
            return []
        
        related = list(self.graph.successors(entity_name)) + list(self.graph.predecessors(entity_name))
        return list(set(related))  # Remove duplicates
    
    def describe(self, entity_name):
        """Describe an entity in the world model."""
        if entity_name in self.variables:
            var = self.variables[entity_name]
            print(f"Variable: {entity_name}")
            print(f"Description: {var['description']}")
            print(f"Value: {var['value']}")
            print(f"Unit: {var['unit']}")
            
            related = self.get_related(entity_name)
            if related:
                print(f"Related to: {', '.join(related)}")
                
        elif entity_name in self.equations:
            eq = self.equations[entity_name]
            print(f"Equation: {entity_name}")
            print(f"Description: {eq['description']}")
            print(f"Formula: {eq['equation']}")
            
            related = self.get_related(entity_name)
            if related:
                print(f"Related to: {', '.join(related)}")
                
        elif entity_name in self.concepts:
            concept = self.concepts[entity_name]
            print(f"Concept: {entity_name}")
            print(f"Description: {concept['description']}")
            
            related = self.get_related(entity_name)
            if related:
                print(f"Related to: {', '.join(related)}")
                
        else:
            print(f"Entity '{entity_name}' not found in the world model.")
    
    def export_to_json(self, filename):
        """Export the world model to a JSON file."""
        import json
        
        # Create a serializable representation of the world model
        data = {
            'variables': self.variables,
            'equations': self.equations,
            'concepts': self.concepts,
            'connections': [{'source': s, 'target': t} for s, t in self.graph.edges()]
        }
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"World model exported to {filename}")
    
    def import_from_json(self, filename):
        """Import the world model from a JSON file."""
        import json
        
        # Read from file
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Clear existing data
        self.graph = nx.DiGraph()
        self.variables = {}
        self.equations = {}
        self.concepts = {}
        
        # Populate the world model
        self.variables = data['variables']
        self.equations = data['equations']
        self.concepts = data['concepts']
        
        # Add nodes to the graph
        for var_name in self.variables:
            self.graph.add_node(var_name, type='variable')
        
        for eq_name in self.equations:
            self.graph.add_node(eq_name, type='equation')
        
        for concept_name in self.concepts:
            self.graph.add_node(concept_name, type='concept')
        
        # Add edges to the graph
        for conn in data['connections']:
            self.graph.add_edge(conn['source'], conn['target'])
        
        print(f"World model imported from {filename}")
    
    def find_path(self, source, target):
        """Find a path between two entities in the world model."""
        if source not in self.graph or target not in self.graph:
            return None
        
        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def get_subgraph(self, entities):
        """Get a subgraph of the world model containing only the specified entities."""
        if not all(entity in self.graph for entity in entities):
            missing = [entity for entity in entities if entity not in self.graph]
            print(f"Warning: The following entities are not in the world model: {missing}")
            entities = [entity for entity in entities if entity in self.graph]
        
        subgraph = self.graph.subgraph(entities)
        return subgraph
    
    def visualize_subgraph(self, entities, figsize=(10, 8)):
        """Visualize a subgraph of the world model."""
        subgraph = self.get_subgraph(entities)
        
        plt.figure(figsize=figsize)
        
        # Define node colors based on type
        node_colors = []
        for node in subgraph.nodes():
            if node in self.variables:
                node_colors.append('skyblue')
            elif node in self.equations:
                node_colors.append('lightgreen')
            else:  # concept
                node_colors.append('salmon')
        
        # Draw the graph
        pos = nx.spring_layout(subgraph, seed=42)
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=500, alpha=0.8)
        nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(subgraph, pos, font_size=10)
        
        # Add a legend
        plt.plot([], [], 'o', color='skyblue', label='Variable')
        plt.plot([], [], 'o', color='lightgreen', label='Equation')
        plt.plot([], [], 'o', color='salmon', label='Concept')
        plt.legend()
        
        plt.axis('off')
        plt.title('Sisyphus Cooling Subgraph')
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create and populate the world model
    world_model = WorldModel()
    
    # Add variables
    world_model.add_variable('RB87_MASS', RB87_MASS, 'kg', 'Mass of a Rubidium-87 atom')
    world_model.add_variable('WAVELENGTH_D2', WAVELENGTH_D2, 'm', 'Wavelength of the D2 transition in Rubidium-87')
    world_model.add_variable('WAVENUMBER_D2', WAVENUMBER_D2, 'm^-1', 'Wavenumber of the D2 transition')
    world_model.add_variable('GAMMA_D2', GAMMA_D2, 's^-1', 'Natural linewidth of the D2 transition')
    world_model.add_variable('RECOIL_ENERGY', RECOIL_ENERGY, 'J', 'Recoil energy from absorbing or emitting a photon')
    world_model.add_variable('RECOIL_TEMPERATURE', RECOIL_TEMPERATURE, 'K', 'Temperature corresponding to the recoil energy')
    world_model.add_variable('DOPPLER_TEMPERATURE', DOPPLER_TEMPERATURE, 'K', 'Doppler cooling limit temperature')
    
    # Add equations
    world_model.add_equation('WAVENUMBER_EQUATION', 'k = 2π/λ', 'Relation between wavenumber and wavelength', 
                             ['WAVENUMBER_D2', 'WAVELENGTH_D2'])
    world_model.add_equation('RECOIL_ENERGY_EQUATION', 'E_r = (ħk)²/(2m)', 'Recoil energy formula', 
                             ['RECOIL_ENERGY', 'WAVENUMBER_D2', 'RB87_MASS'])
    world_model.add_equation('RECOIL_TEMPERATURE_EQUATION', 'T_r = E_r/k_B', 'Recoil temperature formula', 
                             ['RECOIL_TEMPERATURE', 'RECOIL_ENERGY'])
    world_model.add_equation('DOPPLER_TEMPERATURE_EQUATION', 'T_D = ħΓ/(2k_B)', 'Doppler temperature formula', 
                             ['DOPPLER_TEMPERATURE', 'GAMMA_D2'])
    
    # Add concepts
    world_model.add_concept('Sisyphus Cooling', 'A sub-Doppler laser cooling technique where atoms lose energy by climbing potential hills', 
                            ['RECOIL_TEMPERATURE', 'DOPPLER_TEMPERATURE'])
    world_model.add_concept('Optical Lattice', 'A periodic potential created by interfering laser beams', 
                            ['WAVELENGTH_D2', 'WAVENUMBER_D2'])
    world_model.add_concept('Polarization Gradient', 'Spatial variation of light polarization in the optical lattice', 
                            ['Optical Lattice'])
    world_model.add_concept('Optical Pumping', 'Process of transferring atoms between ground state sublevels through absorption and emission cycles', 
                            ['GAMMA_D2', 'Sisyphus Cooling'])
    world_model.add_concept('D2 Transition', 'The 5²S₁/₂ → 5²P₃/₂ transition in Rubidium-87', 
                            ['WAVELENGTH_D2', 'GAMMA_D2'])
    
    # Add additional connections
    world_model.connect('Sisyphus Cooling', 'Optical Lattice')
    world_model.connect('Sisyphus Cooling', 'Polarization Gradient')
    world_model.connect('Sisyphus Cooling', 'Optical Pumping')
    world_model.connect('D2 Transition', 'Optical Pumping')
    
    # Visualize the world model
    world_model.visualize()
    
    # Export to JSON
    world_model.export_to_json('sisyphus_cooling_world_model.json')
