"""
Security Tools Utilities
----------------------
Utility functions for working with security tools,
command execution, and result processing.
"""

import asyncio
import subprocess
import time
from typing import Dict, Any, Optional, List, Union
import logging
import os
import json
import tempfile
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """
    Command execution result container.
    """

    command: str
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time: float = 0.0

    @property
    def success(self) -> bool:
        """Check if the command executed successfully"""
        return self.return_code == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)

    def __str__(self) -> str:
        """String representation"""
        status = "SUCCESS" if self.success else "FAILED"
        return f"Command '{self.command}' {status} in {self.execution_time:.2f}s with return code {self.return_code}"


async def execute_command_async(
    command: str, timeout: int = 60, shell: bool = False
) -> CommandResult:
    """
    Execute a command asynchronously

    Args:
        command: Command to execute
        timeout: Timeout in seconds
        shell: Whether to use shell execution

    Returns:
        Command execution result
    """
    try:
        start_time = time.time()

        # Create subprocess
        process = (
            await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=shell,
            )
            if shell
            else await asyncio.create_subprocess_exec(
                *command.split(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        )

        # Wait for the process to complete with timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout
            )
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            return_code = process.returncode or 0
        except asyncio.TimeoutError:
            # Kill the process if it times out
            try:
                process.kill()
            except Exception:
                pass

            stdout_str = ""
            stderr_str = f"Command timed out after {timeout} seconds"
            return_code = -1

        execution_time = time.time() - start_time

        return CommandResult(
            command=command,
            stdout=stdout_str,
            stderr=stderr_str,
            return_code=return_code,
            execution_time=execution_time,
        )

    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        return CommandResult(
            command=command,
            stdout="",
            stderr=f"Error executing command: {str(e)}",
            return_code=1,
            execution_time=0.0,
        )


def execute_command(
    command: str, timeout: int = 60, shell: bool = False
) -> CommandResult:
    """
    Execute a command synchronously

    Args:
        command: Command to execute
        timeout: Timeout in seconds
        shell: Whether to use shell execution

    Returns:
        Command execution result
    """
    try:
        start_time = time.time()

        # Execute the command
        process = subprocess.Popen(
            command if shell else command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
            text=True,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return_code = process.returncode or 0
        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            stdout, stderr = process.communicate()
            stdout = stdout or ""
            stderr = (
                f"{stderr or ''}\nCommand timed out after {timeout} seconds"
            )
            return_code = -1

        execution_time = time.time() - start_time

        return CommandResult(
            command=command,
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
            execution_time=execution_time,
        )

    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        return CommandResult(
            command=command,
            stdout="",
            stderr=f"Error executing command: {str(e)}",
            return_code=1,
            execution_time=0.0,
        )


async def execute_docker_command(
    docker_image: str, command: str, timeout: int = 60
) -> CommandResult:
    """
    Execute a command in a Docker container

    Args:
        docker_image: Docker image to use
        command: Command to execute in the container
        timeout: Timeout in seconds

    Returns:
        Command execution result
    """
    try:
        # Create a container ID
        container_id = f"nyxtrace-{int(time.time())}"

        # Build the docker command
        docker_command = (
            f"docker run --name {container_id} --rm {docker_image} {command}"
        )

        # Execute the command
        result = await execute_command_async(
            docker_command, timeout, shell=True
        )

        # Clean up any remaining containers
        try:
            await execute_command_async(
                f"docker rm -f {container_id}", 10, shell=True
            )
        except Exception:
            pass

        return result

    except Exception as e:
        logger.error(f"Error executing Docker command: {str(e)}")
        return CommandResult(
            command=f"docker run {docker_image} {command}",
            stdout="",
            stderr=f"Error executing Docker command: {str(e)}",
            return_code=1,
            execution_time=0.0,
        )


async def execute_remote_command(
    host: str,
    user: str,
    command: str,
    key_path: Optional[str] = None,
    password: Optional[str] = None,
    timeout: int = 60,
) -> CommandResult:
    """
    Execute a command on a remote host via SSH

    Args:
        host: Remote host
        user: Remote user
        command: Command to execute
        key_path: Path to SSH key
        password: SSH password
        timeout: Timeout in seconds

    Returns:
        Command execution result
    """
    try:
        # Build the SSH command
        ssh_command = f"ssh"

        if key_path:
            ssh_command += f" -i {key_path}"

        ssh_command += f" {user}@{host} '{command}'"

        # Execute the command
        if password:
            # Use sshpass if password is provided
            ssh_command = f"sshpass -p '{password}' {ssh_command}"

        return await execute_command_async(ssh_command, timeout, shell=True)

    except Exception as e:
        logger.error(f"Error executing remote command: {str(e)}")
        return CommandResult(
            command=f"ssh {user}@{host} '{command}'",
            stdout="",
            stderr=f"Error executing remote command: {str(e)}",
            return_code=1,
            execution_time=0.0,
        )


def parse_nmap_output(nmap_output: str) -> Dict[str, Any]:
    """
    Parse nmap output into structured data

    Args:
        nmap_output: Nmap command output

    Returns:
        Structured data from nmap output
    """
    result = {"hosts": [], "ports": []}

    # Extract hosts
    import re

    host_matches = re.finditer(r"Nmap scan report for (.*)", nmap_output)
    for match in host_matches:
        host = match.group(1).strip()
        result["hosts"].append(host)

    # Extract open ports
    port_matches = re.finditer(
        r"(\d+)\/(\w+)\s+(\w+)\s+(\w+)(?:\s+(.*))?", nmap_output
    )
    for match in port_matches:
        port = {
            "port": int(match.group(1)),
            "protocol": match.group(2),
            "state": match.group(3),
            "service": match.group(4),
            "details": (
                match.group(5)
                if len(match.groups()) > 4 and match.group(5)
                else ""
            ),
        }
        result["ports"].append(port)

    return result


def parse_nikto_output(nikto_output: str) -> Dict[str, Any]:
    """
    Parse nikto output into structured data

    Args:
        nikto_output: Nikto command output

    Returns:
        Structured data from nikto output
    """
    result = {"target": "", "vulnerabilities": []}

    # Extract target
    import re

    target_match = re.search(r"- Target: (.*)", nikto_output)
    if target_match:
        result["target"] = target_match.group(1).strip()

    # Extract vulnerabilities
    vuln_matches = re.finditer(r"\+ (.*): (.*)", nikto_output)
    for match in vuln_matches:
        vulnerability = {
            "id": match.group(1).strip(),
            "description": match.group(2).strip(),
        }
        result["vulnerabilities"].append(vulnerability)

    return result


def check_tool_installed(tool_name: str) -> bool:
    """
    Check if a security tool is installed

    Args:
        tool_name: Name of the tool to check

    Returns:
        True if installed, False otherwise
    """
    try:
        result = execute_command(f"which {tool_name}", timeout=5)
        return result.success
    except Exception:
        return False


def download_kali_tool_image(tool_name: str) -> str:
    """
    Download a Kali Linux tool Docker image

    Args:
        tool_name: Name of the tool

    Returns:
        Docker image name
    """
    try:
        # Create a custom image name
        image_name = f"nyxtrace-{tool_name.replace(' ', '-')}"

        # Check if the image already exists
        check_result = execute_command(
            f"docker image inspect {image_name}", timeout=10
        )

        if check_result.success:
            return image_name

        # Create a Dockerfile
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".Dockerfile"
        ) as f:
            dockerfile_path = f.name
            f.write(
                f"""FROM kalilinux/kali-rolling
RUN apt-get update && apt-get install -y {tool_name}
CMD ["/bin/bash"]
"""
            )

        # Build the Docker image
        build_result = execute_command(
            f"docker build -t {image_name} -f {dockerfile_path} .", timeout=300
        )

        # Clean up
        os.unlink(dockerfile_path)

        if build_result.success:
            return image_name
        else:
            # Return the default Kali image if build fails
            return "kalilinux/kali-rolling"

    except Exception as e:
        logger.error(f"Error downloading Kali tool image: {str(e)}")
        return "kalilinux/kali-rolling"


def format_command_output(
    command_result: CommandResult, format_type: str = "plain"
) -> str:
    """
    Format command output for display

    Args:
        command_result: Command execution result
        format_type: Output format (plain, html, markdown)

    Returns:
        Formatted output
    """
    if format_type == "plain":
        output = f"Command: {command_result.command}\n"
        output += f"Return Code: {command_result.return_code}\n"
        output += f"Execution Time: {command_result.execution_time:.2f}s\n"
        output += f"\n--- STDOUT ---\n{command_result.stdout}\n"
        output += f"\n--- STDERR ---\n{command_result.stderr}\n"
        return output

    elif format_type == "html":
        success_class = (
            "text-success" if command_result.success else "text-danger"
        )
        html = f"""
        <div class="command-result">
            <div class="command"><strong>Command:</strong> {command_result.command}</div>
            <div class="status"><strong>Status:</strong> <span class="{success_class}">{'SUCCESS' if command_result.success else 'FAILED'}</span></div>
            <div class="return-code"><strong>Return Code:</strong> {command_result.return_code}</div>
            <div class="execution-time"><strong>Execution Time:</strong> {command_result.execution_time:.2f}s</div>
            
            <div class="output-section">
                <div class="section-heading">STDOUT</div>
                <pre class="output">{command_result.stdout}</pre>
            </div>
            
            <div class="output-section">
                <div class="section-heading">STDERR</div>
                <pre class="output">{command_result.stderr}</pre>
            </div>
        </div>
        """
        return html

    elif format_type == "markdown":
        status = "✅ SUCCESS" if command_result.success else "❌ FAILED"
        markdown = f"""
### Command Execution Result

**Command:** `{command_result.command}`
**Status:** {status}
**Return Code:** {command_result.return_code}
**Execution Time:** {command_result.execution_time:.2f}s

#### STDOUT
```
{command_result.stdout}
```

#### STDERR
```
{command_result.stderr}
```
        """
        return markdown

    # Default to plain
    return f"{command_result.stdout}\n{command_result.stderr}"


def save_command_result(
    command_result: CommandResult, output_file: str, format_type: str = "json"
) -> bool:
    """
    Save command result to a file

    Args:
        command_result: Command execution result
        output_file: Output file path
        format_type: Output format (json, text, csv)

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_file, "w") as f:
            if format_type == "json":
                # Save as JSON
                json.dump(command_result.to_dict(), f, indent=2)
            elif format_type == "csv":
                # Save as CSV
                f.write(f"Command,Return Code,Execution Time,Success\n")
                f.write(
                    f'"{command_result.command}",{command_result.return_code},{command_result.execution_time},{command_result.success}\n'
                )
                f.write(f"\nSTDOUT:\n{command_result.stdout}\n")
                f.write(f"\nSTDERR:\n{command_result.stderr}\n")
            else:
                # Save as plain text
                f.write(format_command_output(command_result, "plain"))

        return True

    except Exception as e:
        logger.error(f"Error saving command result: {str(e)}")
        return False


def build_command(
    tool_name: str,
    options: Dict[str, Any],
    command_template: Optional[str] = None,
) -> str:
    """
    Build a command for a security tool

    Args:
        tool_name: Name of the tool
        options: Command options
        command_template: Optional command template

    Returns:
        Full command string
    """
    # Common templates for popular tools
    templates = {
        "nmap": "nmap {target} {options}",
        "nikto": "nikto -h {host} {options}",
        "sqlmap": "sqlmap -u {url} {options}",
        "gobuster": "gobuster dir -u {url} -w {wordlist} {options}",
    }

    # Use provided template, tool-specific template, or default format
    template = command_template or templates.get(
        tool_name.lower(), "{tool} {options}"
    )

    # Replace {tool} with the actual tool name if present
    template = template.replace("{tool}", tool_name)

    # Build options string
    options_str = ""
    for key, value in options.items():
        if key in template:
            # If the key is in the template, it will be replaced directly
            continue

        if value is True:
            # Boolean flag
            options_str += f" --{key}"
        elif value is False:
            # Skip false boolean flags
            continue
        elif value is not None:
            # Regular option
            options_str += f" --{key} {value}"

    # Replace {options} with the built options string
    template = template.replace("{options}", options_str.strip())

    # Replace any remaining placeholders with their values from the options dict
    for key, value in options.items():
        placeholder = f"{{{key}}}"
        if placeholder in template and value is not None:
            template = template.replace(placeholder, str(value))

    return template.strip()
