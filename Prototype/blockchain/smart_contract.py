"""
PQBFL Smart Contract Implementation
Implements the smart contract logic from Algorithm 1 in the paper
"""

from typing import Dict, List, Optional
import time

from .blockchain import Blockchain
from .transactions import (
    RegisterProjectTx,
    RegisterClientTx,
    PublishTaskTx,
    UpdateModelTx,
    FeedbackModelTx,
    TerminateProjectTx
)


class PQBFLSmartContract:
    """
    Smart contract for PQBFL protocol
    Manages projects, clients, tasks, updates, and feedback
    """
    
    def __init__(self, blockchain: Blockchain, gas_costs: Dict[str, int]):
        self.blockchain = blockchain
        self.gas_costs = gas_costs
        
        # Contract state
        self.projects: Dict[str, dict] = {}
        self.clients: Dict[str, dict] = {}
        self.tasks: Dict[str, dict] = {}
        self.updates: Dict[str, List[dict]] = {}
        self.feedbacks: Dict[str, List[dict]] = {}
        self.done: Dict[str, bool] = {}
        
        # Required deposit for project registration
        self.required_deposit = 1000.0
    
    def register_project(
        self,
        sender: str,
        project_id: str,
        n_clients: int,
        h_initial_model: str,
        h_server_keys: str,
        deposit: float
    ) -> tuple[bool, int, Optional[str]]:
        """
        Register a new FL project
        Returns: (success, gas_used, error_message)
        """
        gas_used = self.gas_costs.get('register_project', 250000)
        
        # Validate deposit
        if deposit < self.required_deposit:
            return False, gas_used, "Insufficient deposit"
        
        # Check if project already exists and is not done
        if project_id in self.projects and not self.done.get(project_id, False):
            return False, gas_used, "Project already exists"
        
        # Create project
        self.projects[project_id] = {
            'project_id': project_id,
            'n_clients': n_clients,
            'server_address': sender,
            'h_initial_model': h_initial_model,
            'h_server_keys': h_server_keys,
            'deposit': deposit,
            'registered_clients': 0,
            'timestamp': time.time()
        }
        
        self.done[project_id] = False
        
        # Create and add transaction to blockchain
        tx = RegisterProjectTx(
            sender=sender,
            project_id=project_id,
            n_clients=n_clients,
            h_initial_model=h_initial_model,
            h_server_keys=h_server_keys,
            deposit=deposit
        )
        
        self.blockchain.add_transaction(tx)
        
        return True, gas_used, None
    
    def register_client(
        self,
        sender: str,
        project_id: str,
        h_client_ecdh_key: str,
        initial_score: int = 0
    ) -> tuple[bool, int, Optional[str]]:
        """
        Register a client to a project
        Returns: (success, gas_used, error_message)
        """
        gas_used = self.gas_costs.get('register_client', 75000)
        
        # Check if project exists
        if project_id not in self.projects:
            return False, gas_used, "Project does not exist"
        
        project = self.projects[project_id]
        
        # Check if project has capacity
        if project['registered_clients'] >= project['n_clients']:
            return False, gas_used, "Project is full"
        
        # Check if client already registered
        if sender in self.clients and self.clients[sender].get('project_id') == project_id:
            return False, gas_used, "Client already registered"
        
        # Register client
        self.clients[sender] = {
            'address': sender,
            'project_id': project_id,
            'h_ecdh_key': h_client_ecdh_key,
            'score': initial_score,
            'timestamp': time.time()
        }
        
        # Update project client count
        project['registered_clients'] += 1
        
        # Create and add transaction
        tx = RegisterClientTx(
            sender=sender,
            project_id=project_id,
            h_client_ecdh_key=h_client_ecdh_key,
            initial_score=initial_score
        )
        
        self.blockchain.add_transaction(tx)
        
        return True, gas_used, None
    
    def publish_task(
        self,
        sender: str,
        project_id: str,
        task_id: str,
        round_number: int,
        h_global_model: str,
        h_server_keys: Optional[str],
        deadline: float,
        n_clients_required: int
    ) -> tuple[bool, int, Optional[str]]:
        """
        Publish a new training task
        Returns: (success, gas_used, error_message)
        """
        gas_used = self.gas_costs.get('publish_task', 260000)
        
        # Verify sender is the project server
        if project_id not in self.projects:
            return False, gas_used, "Project does not exist"
        
        if self.projects[project_id]['server_address'] != sender:
            return False, gas_used, "Only project server can publish tasks"
        
        # Create task
        self.tasks[task_id] = {
            'task_id': task_id,
            'project_id': project_id,
            'round_number': round_number,
            'h_global_model': h_global_model,
            'h_server_keys': h_server_keys,
            'deadline': deadline,
            'n_clients_required': n_clients_required,
            'server_address': sender,
            'timestamp': time.time()
        }
        
        # Initialize updates list for this task
        self.updates[task_id] = []
        
        # Create and add transaction
        tx = PublishTaskTx(
            sender=sender,
            project_id=project_id,
            task_id=task_id,
            round_number=round_number,
            h_global_model=h_global_model,
            h_server_keys=h_server_keys,
            deadline=deadline,
            n_clients_required=n_clients_required
        )
        
        self.blockchain.add_transaction(tx)
        
        return True, gas_used, None
    
    def update_model(
        self,
        sender: str,
        project_id: str,
        task_id: str,
        round_number: int,
        h_local_model: str,
        h_client_keys: Optional[str]
    ) -> tuple[bool, int, Optional[str]]:
        """
        Submit a local model update
        Returns: (success, gas_used, error_message)
        """
        gas_used = self.gas_costs.get('update_model', 235000)
        
        # Verify task exists
        if task_id not in self.tasks:
            return False, gas_used, "Task does not exist"
        
        # Verify client is registered
        if sender not in self.clients:
            return False, gas_used, "Client not registered"
        
        # Verify client belongs to project
        if self.clients[sender]['project_id'] != project_id:
            return False, gas_used, "Client not in this project"
        
        # Check deadline
        task = self.tasks[task_id]
        if time.time() > task['deadline']:
            return False, gas_used, "Task deadline passed"
        
        # Record update
        update = {
            'task_id': task_id,
            'client_address': sender,
            'round_number': round_number,
            'h_local_model': h_local_model,
            'h_client_keys': h_client_keys,
            'timestamp': time.time()
        }
        
        self.updates[task_id].append(update)
        
        # Create and add transaction
        tx = UpdateModelTx(
            sender=sender,
            project_id=project_id,
            task_id=task_id,
            round_number=round_number,
            h_local_model=h_local_model,
            h_client_keys=h_client_keys
        )
        
        self.blockchain.add_transaction(tx)
        
        return True, gas_used, None
    
    def provide_feedback(
        self,
        sender: str,
        project_id: str,
        task_id: str,
        round_number: int,
        client_address: str,
        score: int,
        terminate: bool = False
    ) -> tuple[bool, int, Optional[str]]:
        """
        Provide feedback on a client's model update
        Returns: (success, gas_used, error_message)
        """
        gas_used = self.gas_costs.get('feedback_model', 215000)
        
        # Verify sender is project server
        if project_id not in self.projects:
            return False, gas_used, "Project does not exist"
        
        if self.projects[project_id]['server_address'] != sender:
            return False, gas_used, "Only project server can provide feedback"
        
        # Verify task exists
        if task_id not in self.tasks:
            return False, gas_used, "Task does not exist"
        
        # Update client score
        self._update_client_score(client_address, score)
        
        # Record feedback
        if task_id not in self.feedbacks:
            self.feedbacks[task_id] = []
        
        feedback = {
            'task_id': task_id,
            'round_number': round_number,
            'client_address': client_address,
            'score': score,
            'terminate': terminate,
            'timestamp': time.time()
        }
        
        self.feedbacks[task_id].append(feedback)
        
        # Create and add transaction
        tx = FeedbackModelTx(
            sender=sender,
            project_id=project_id,
            task_id=task_id,
            round_number=round_number,
            client_address=client_address,
            score=score,
            terminate=terminate
        )
        
        self.blockchain.add_transaction(tx)
        
        return True, gas_used, None
    
    def _update_client_score(self, client_address: str, score_delta: int):
        """Update a client's reputation score"""
        if client_address in self.clients:
            current_score = self.clients[client_address]['score']
            new_score = max(0, current_score + score_delta)  # Min score is 0
            self.clients[client_address]['score'] = new_score
    
    def finish_project(
        self,
        sender: str,
        project_id: str,
        final_task_id: str,
        final_round: int
    ) -> tuple[bool, int, Optional[str]]:
        """
        Mark a project as complete
        Returns: (success, gas_used, error_message)
        """
        gas_used = 100000  # Termination gas cost
        
        # Verify sender is project server
        if project_id not in self.projects:
            return False, gas_used, "Project does not exist"
        
        if self.projects[project_id]['server_address'] != sender:
            return False, gas_used, "Only project server can finish project"
        
        # Mark project as done
        self.done[project_id] = True
        
        # Create and add transaction
        tx = TerminateProjectTx(
            sender=sender,
            project_id=project_id,
            task_id=final_task_id,
            final_round=final_round
        )
        
        self.blockchain.add_transaction(tx)
        
        return True, gas_used, None
    
    def get_project(self, project_id: str) -> Optional[dict]:
        """Get project information"""
        return self.projects.get(project_id)
    
    def get_client(self, client_address: str) -> Optional[dict]:
        """Get client information"""
        return self.clients.get(client_address)
    
    def get_task(self, task_id: str) -> Optional[dict]:
        """Get task information"""
        return self.tasks.get(task_id)
    
    def get_task_updates(self, task_id: str) -> List[dict]:
        """Get all updates for a task"""
        return self.updates.get(task_id, [])
    
    def get_task_feedbacks(self, task_id: str) -> List[dict]:
        """Get all feedbacks for a task"""
        return self.feedbacks.get(task_id, [])
    
    def get_client_score(self, client_address: str) -> int:
        """Get a client's reputation score"""
        if client_address in self.clients:
            return self.clients[client_address]['score']
        return 0
