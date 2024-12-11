# API Reference Documentation

The Qubot-One API provides a set of endpoints and methods for interacting with the system. This document outlines the available APIs, their parameters, and usage examples.

## Base URL

The base URL for accessing the Qubot-One API is:

http://localhost:5000/api


## Authentication

Most API endpoints require authentication. Use the following method to obtain a token:

### POST /auth/login

- **Description**: Authenticate a user and obtain a token.
- **Request Body**:
  ```json
  1 {
  2   "username": "your_username",
  3   "password": "your_password"
  4 }
  ```

**Response**
  ```json
  1 {
  2   "token": "your_jwt_token"
  3 }
  ```

### Endpoints
- **GET** /qubots
- **Description**: Retrieve a list of all Qubots.
- **Headers**:
- **Authorization**: Bearer your_jwt_token
**Response**:
  ```json
  1 [
  2   {
  3     "id": 1,
  4     "name": "Qubot Alpha",
  5     "status": "active"
  6   },
  7   {
  8     "id": 2,
  9     "name": "Qubot Beta",
  10     "status": "inactive"
  11   }
  12 ]
  ```

## POST /qubots
- **Description**: Create a new Qubot.
- **Headers**:
- **Authorization**: Bearer your_jwt_token
- **Request Body**:
  ```json
  1 {
  2   "name": "New Qubot",
  3   "type": "standard"
  4 }
  ```

**Response**:
  ```json
  1 {
  2   "id": 3,
  3   "name": "New Qubot",
  4   "status": "active"
  5 }
  ```

## GET /qubots/{id}
- **Description**: Retrieve details of a specific Qubot.
- **Parameters**:
- **id**: The ID of the Qubot.
- **Headers**:
- **Authorization**: Bearer your_jwt_token
**Response**:
  ```json
  1 {
  2   "id": 1,
  3   "name": "Qubot Alpha",
  4   "status": "active",
  5   "type": "standard",
  6   "last_active": "2023-10-01T12:00:00Z"
  7 }
  ```

## Error Handling
The API uses standard HTTP status codes to indicate the success or failure of a request. Common error responses include:

- 400 Bad Request: The request was invalid or cannot be processed.
- 401 Unauthorized: Authentication failed or token is missing/invalid.
- 404 Not Found: The requested resource could not be found.
- 500 Internal Server Error: An unexpected error occurred on the server.

## Conclusion
This API reference provides a comprehensive overview of the available endpoints and their usage. For further details or updates, please refer to the official documentation or the source code.

