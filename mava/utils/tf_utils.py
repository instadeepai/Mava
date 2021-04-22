from acme.tf import utils as tf2_utils
from acme import types as acme_types

def batch_to_sequence(data: acme_types.NestedTensor) -> acme_types.NestedTensor:
  """Converts data between sequence-major and batch-major format."""

  print("Actions before: ", data[1]["walker_0"])
  coverted_data = []

  coverted_data = tf2_utils.batch_to_sequence(data[0:2] + data[3:])

  # for entry in data[0:2]:
  #   conv_dict = {}
  #   for key, value in entry.items():
  #     print("Before: ", value)
  #
  #     print("Value type: ", type(value))
  #
  #     conv_dict[key] = tf2_utils.batch_to_sequence([value])[0]
  #     print("After: ", conv_dict[key])
  #
  #   coverted_data.append(conv_dict)

  print("Actions after: ", coverted_data[1]["walker_0"])
  exit()
  return