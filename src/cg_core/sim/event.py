from __future__ import annotations
from typing import Any, Callable, Optional, Dict


class Event:
    """
    Simulation event
    """

    def __init__(self,
                 event_id: int,
                 event_name: str,
                 timestamp_start: float,
                 timestamp_end: float,
                 callback: Callable[[Optional[Any]], None],
                 payload: Optional[Dict] = None
                 ) -> None:
        """
        :param event_id: Unique id to track this event in simulation run
        :param event_name: Event name
        :param timestamp_start: Timestamp when this event has been added to the queue
        :param timestamp_end: Timestamp when this event is completed
        :param callback: The function to be called when this event is finished
        :param payload: Optional additional data, to pass to the callback
        """

        self.event_id: int = event_id
        self.event_name: str = event_name
        self.timestamp_start: float = timestamp_start
        self.timestamp_end: float = timestamp_end
        self.callback: Callable[[Optional[Any]], None] = callback
        self.payload: Optional[Dict] = payload
        return

    def __lt__(self, other: Event):
        """
        Guarantees the total ordering of events
        """

        if self.timestamp_end < other.timestamp_end:
            return True
        else:
            return self.event_id < other.event_id

    def retire(self) -> None:
        """
        Must be called when this event is popped from the event queue
        """
        self.callback(self.payload)
