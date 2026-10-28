# pylint: disable=line-too-long
"""
Module for configuring and setting up logging.
"""

import json
import logging
import logging.config
import os
import textwrap
from typing import Optional, cast


class CustomLogger(logging.Logger):
    """
    A custom logger that includes a method to add dividers.

    Attributes
    ----------
    - name: str
        The name of the logger.
    - level: int
        The logging level (e.g., logging.DEBUG, logging.INFO).

    Features
    ----------
    - Supports multi-line messages
    - Allows message truncation (if allow_wrap=False)
    - Adds visual dividers for readability
    - Supports colored console output for different log levels
    - Can write logs to both console and file
    """

    def add_spacer(
        self,
        level: int = logging.INFO,
        lines: int = 1,
    ):
        """
        Add empty lines as spacers in the log.

        Parameters
        ----------
        level: int
            The logging level at which the spacer should be logged (e.g., logging.DEBUG, logging.INFO).
        lines: int
            The number of empty lines to insert in the log.
        """
        # Ensure the number of lines is at least 1
        lines = max(lines, 1)

        for _ in range(lines):
            self.log(level, " ")  # Log an empty line

    def add_divider(
        self,
        level: int = logging.INFO,
        length: int = 10,
        border: str = "+",
        fill: str = "=",
    ):
        """
        Add a divider line to the log at the specified level.

        Parameters
        ----------
        level: int
            The logging level at which the divider should be logged (e.g., logging.DEBUG, logging.INFO).
        length: int
            The number of fill characters in the divider.
        border: str
            The character used for the border of the divider.
        fill: str
            The character used to fill the divider.
        """
        # Ensure the total length is sufficient for the borders and spaces
        length = max(length, 3)
        divider = f"{border}{fill * (length-2)}{border}"
        self.log(level, divider)

    def log_with_borders(
        self,
        level: int,
        message: str,
        border: str = "|",
        length: int = 50,
    ):
        """
        Log a message with borders around it, truncating the message if it exceeds the specified length.
        Adds a space between the border and the message.

        Parameters
        ----------
        level: int
            The logging level at which the message should be logged (e.g., logging.DEBUG, logging.INFO).
        message: str
            The message to log.
        border: str
            The character used for the borders.
        length: int
            The total length of the formatted message, including borders and spaces.
        """
        # Ensure the total length is sufficient for the borders and spaces
        length = max(
            length, 5
        )  # Minimum length to accommodate borders, spaces, and at least one character

        # Calculate the maximum length for the message
        max_message_length = length - 4  # Subtract 4 for borders and spaces

        # Step 1: Split message by newline characters
        message_lines = message.splitlines()

        # Step 2: Wrap each line separately (to prevent cutting in the middle of a paragraph)
        wrapped_lines = []
        for line in message_lines:
            wrapped_lines.extend(
                textwrap.wrap(line, width=max_message_length) or [""]
            )  # Ensure empty lines are preserved

        # Step 3: Format and log each wrapped line with borders
        for line in wrapped_lines:
            formatted_message = f"{border} {line.ljust(max_message_length)} {border}"
            self.log(level, formatted_message)

    def log_title(
        self,
        title: str,
        level: int = logging.INFO,
        length: int = 40,
        border: str = "#",
    ):
        """
        Logs a formatted title with a surrounding border to visually separate log sections.

        This method helps improve log readability by formatting important log sections
        with a clear visual distinction.

        Parameters
        ----------
        title: str
            The title text to log.
        level: int
            The logging level at which the title should be logged (e.g., logging.DEBUG, logging.INFO).
        length: int
            The number of border characters before and after the title.
            The total width of the formatted line will be `2 * length + len(title) + 2` (including spaces).
        border: str
            The character used for the borders.
        """
        # Ensure title fits within the specified length
        formatted_title = f"{border * length} {title} {border * length}"

        # Log the formatted title
        self.log(level, formatted_title)


def setup_logging(
    input_config_file: str = "src/python/general_logging.json",
    logger_name: str = __name__,
    handler_name: str = "general",
    output_log_path: Optional[str] = None,
) -> CustomLogger:
    """
    Load and apply logging configuration from a JSON configuration file.

    Parameters
    ----------
    input_config_file: str
        Path to the logging configuration file.
    logger_name: str
        Name of the logger.
    handler_name: str
        Name of the handler.
    output_log_path: Optional[str]
        Path to the output log file.

    Returns
    ----------
    CustomLogger
        The configured logger.
    """

    # Set the global Logger class to CustomLogger
    logging.setLoggerClass(CustomLogger)

    # Set up temporary logging configuration with formatted output
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)-8s -",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create a temporary logger for logging setup issues
    temp_logger = logging.getLogger("setup_logging_temp_logger")
    temp_logger.setLevel(logging.DEBUG)

    try:
        # Attempt to open and read the JSON configuration file
        with open(input_config_file, "r", encoding="utf-8") as file:
            # Load the JSON file content into a dictionary
            config_dict = json.load(file)

            # Dynamically adjust the output log path if provided
            if output_log_path:
                if handler_name in config_dict.get("handlers", {}):
                    config_dict["handlers"][handler_name]["filename"] = output_log_path
                    temp_logger.debug(
                        "Handler '%s' filename dynamically set to: %s",
                        handler_name,
                        output_log_path,
                    )
                else:
                    raise KeyError(
                        f"Handler '{handler_name}' not found in configuration."
                    )

            # Ensure the directory for the log file exists
            if output_log_path:
                log_dir = os.path.dirname(output_log_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                    temp_logger.debug("Created directory for log file: %s", log_dir)

            # Apply the logging configuration
            logging.config.dictConfig(config_dict)

            # Get the logger
            logger = cast(CustomLogger, logging.getLogger(logger_name))

            return logger

    # Catch errors that might occur during file opening or reading
    except (PermissionError, FileNotFoundError, json.JSONDecodeError) as file_error:
        # Log detailed error message
        temp_logger.error(
            "File operation error: %s. Please check the following:\n"
            "- Does the configuration file exist? Path: %s\n"
            "- Does the file have the correct read permissions?\n"
            "- Is the configuration file a valid JSON format?",
            file_error,
            input_config_file,
        )
        raise

    # Catch errors that might occur during the configuration process
    except (ValueError, KeyError) as config_error:
        # Log detailed error message
        temp_logger.error(
            "Configuration error: %s. Please check your JSON configuration file for:\n"
            "- Missing required fields (e.g., handlers, loggers).\n"
            "- Incorrect field values.",
            config_error,
        )
        raise

    # Catch any other unforeseen exceptions
    except Exception as unexpected_error:
        # Log detailed error message
        temp_logger.error(
            "An unexpected error occurred: %s.\n"
            "Please verify the configuration file, input parameters, and execution environment for potential issues.",
            unexpected_error,
        )
        raise
