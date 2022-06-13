import hashlib
from typing import List


class HashableHooks:
    initial_token_value = "initial_token_value"

    def reset_token(self) -> None:
        """Reset token to initial value"""
        self.token = self.initial_token_value

    def hash_token(self, hash_by: str) -> None:
        """Use 'hash_by' to hash the given string token"""
        if not hasattr(self, "token"):
            raise Exception(
                "Initial token needs to be created using HashableHooks.reset_token()"
            )
        self.token = self._hash_function(self.token, hash_by)

    @staticmethod
    def _hash_function(token: str, hash_by: str) -> str:
        """Use 'hash_by' to hash the given string token"""
        return hashlib.md5((token + hash_by).encode()).hexdigest()

    @staticmethod
    def get_final_token_value(method_names: List[str]) -> str:
        """Get the value of initial token after it is hashed by the method names"""
        token = HashableHooks.initial_token_value
        for method_name in method_names:
            token = HashableHooks._hash_function(token, method_name)
        return token

    # init hooks
    def on_parameter_server_init_start(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_init_start")

    def on_parameter_server_init(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_init")

    def on_parameter_server_init_checkpointer(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_init_checkpointer")

    def on_parameter_server_init_end(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_init_end")

    # get_parameters hooks
    def on_parameter_server_get_parameters_start(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_get_parameters_start")

    def on_parameter_server_get_parameters(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_get_parameters")

    def on_parameter_server_get_parameters_end(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_get_parameters_end")

    # set_parameters hooks
    def on_parameter_server_set_parameters_start(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_set_parameters_start")

    def on_parameter_server_set_parameters(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_set_parameters")

    def on_parameter_server_set_parameters_end(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_set_parameters_end")

    # add_to_parameters hooks
    def on_parameter_server_add_to_parameters_start(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_add_to_parameters_start")

    def on_parameter_server_add_to_parameters(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_add_to_parameters")

    def on_parameter_server_add_to_parameters_end(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_add_to_parameters_end")

    # step hooks
    def on_parameter_server_run_loop_start(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_run_loop_start")

    def on_parameter_server_run_loop_checkpoint(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_run_loop_checkpoint")

    def on_parameter_server_run_loop(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_run_loop")

    def on_parameter_server_run_loop_termination(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_run_loop_termination")

    def on_parameter_server_run_loop_end(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_run_loop_end")

    # run hooks
    def on_parameter_server_run_start(self) -> None:
        """Override hook to update token using the method name"""
        self.hash_token("on_parameter_server_run_start")
