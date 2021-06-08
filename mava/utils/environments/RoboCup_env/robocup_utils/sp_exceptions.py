# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# type: ignore


class SoccerServerError(Exception):
    """
    Represents an error message returned by the soccer server.
    """


class SoccerServerWarning(Exception):
    """
    Represents a warning message returned by the soccer server.
    """


class MessageTypeError(Exception):
    """
    An exception for an unknown message type received from the soccer server.
    """


class AgentAlreadyPlayingError(Exception):
    """
    Raised when a user calls an agent's play method after it has already started
    playing.
    """


class ObjectTypeError(Exception):
    """
    Raised when an unknown object type is encountered in a sense message.
    """


class AgentConnectionStateError(Exception):
    """
    Raised when methods are called at an inappropriate time relative to the
    connection state of the agent object.
    """
