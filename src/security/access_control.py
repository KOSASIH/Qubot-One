# src/security/access_control.py

class AccessControl:
    def __init__(self):
        """Initialize the access control manager."""
        self.roles = {}
        self.permissions = {}

    def add_role(self, role):
        """Add a role to the access control system.

        Args:
            role (str): The name of the role to add.
        """
        if role not in self.roles:
            self.roles[role] = []
            print(f"Role added: {role}")

    def add_permission(self, permission):
        """Add a permission to the access control system.

        Args:
            permission (str): The name of the permission to add.
        """
        if permission not in self.permissions:
            self.permissions[permission] = []
            print(f"Permission added: {permission}")

    def assign_permission_to_role(self, role, permission):
        """Assign a permission to a role.

        Args:
            role (str): The role to assign the permission to.
            permission (str): The permission to assign.
        """
        if role in self.roles and permission in self.permissions:
            self.roles[role].append(permission)
            self.permissions[permission].append(role)
            print(f"Assigned permission '{permission}' to role '{role}'")

    def check_access(self, role, permission):
        """Check if a role has a specific permission.

        Args:
            role (str): The role to check.
            permission (str): The permission to check.

        Returns:
            bool: True if the role has the permission, False otherwise.
        """
        has_access = permission in self.roles.get(role, [])
        print(f"Access check for role '{role}' and permission '{permission}': {has_access}")
        return has_access
