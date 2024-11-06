import torch
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

WAYMO_METRIC_NAMES = [
    'metametric', 'linear_acceleration_likelihood', 'time_to_collision_likelihood',
    'offroad_indication_likelihood', 'min_average_displacement_error', 'linear_speed_likelihood',
    'distance_to_road_edge_likelihood', 'distance_to_nearest_object_likelihood', 
    'collision_indication_likelihood', 'average_displacement_error', 
    'angular_speed_likelihood', 'angular_acceleration_likelihood'
]

class WaymoMetrics(torchmetrics.Metric):
    def __init__(self):
        """
        Initializes the BatchMetrics class with a TensorBoard writer and metric names.
        
        Parameters:
        - writer: TensorBoard SummaryWriter instance for logging metrics.
        - metric_names: List of metric names to track and log.
        """
        super().__init__(dist_sync_on_step=False)
        self.metric_names = WAYMO_METRIC_NAMES

        # Add the metrics as attributes to the class
        for name in self.metric_names:
            setattr(self, name, torchmetrics.MeanMetric().to(self.device))
    def update(self, scenario_metrics):
        """
        Updates each metric with the new batch data.
        
        Parameters:
        - scenario_metrics: A list of metric objects for each sample in the batch.
        """

        # Update each MeanMetric with the batch data
        for name in self.metric_names:
            value = getattr(scenario_metrics, name)
            values_tensor = torch.tensor(value, device=self.device)
            getattr(self, name).update(values_tensor)


    def compute(self, step, writer):
        """
        Logs the average and histogram for each metric in the batch to TensorBoard.
        
        Parameters:
        - step: The current step or epoch for logging in TensorBoard.
        """
        for name in self.metric_names:
            metric = getattr(self, name)
            # Log the computed average to TensorBoard
            avg_value = metric.compute()
            writer.add_scalar(f'metrics/average_{name}', avg_value, step)


    def reset(self):
        """
        Resets all metrics to start fresh for the next batch.
        """
        for name in self.metric_names:
            getattr(self, name).reset()
