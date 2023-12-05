# API Documentation

This repository contains a Node.js and Express API for managing user data. The API includes endpoints for retrieving user information and updating user data.

## Installation

Follow these steps to set up and run the API:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yunusarridwan/fitfriend-api.git
   ```

2. Navigate to the project directory:

   ```bash
   cd fitfriend-api
   ```

3. Install the required dependencies:

   ```bash
   npm install
   ```

4. Create a `.env` file in the root directory and add the necessary environment variables:

   ```
   PORT=8080
   MONGODB_URI=mongodb://fitfriend:fitfriend123@clusterproject-shard-00-00.rqlvg.mongodb.net:27017,clusterproject-shard-00-01.rqlvg.mongodb.net:27017,clusterproject-shard-00-02.rqlvg.mongodb.net:27017/fitfriend?replicaSet=atlas-t63wbg-shard-0&ssl=true&authSource=admin
   ```

5. Start the server:

   ```bash
   npm start
   ```

   The API will be accessible at `https://fitfriend-api-lgc5c6xm5a-as.a.run.app:8080`.

## Endpoints

### 1. Get User Data

- **Endpoint:** `PUT /user`
- **Middleware:** `getResponseAuth`, `saveDataUser`
- **Description:** Retrieves data for the authenticated user.
- **Request Body:** None
- **Response:**
  ```json
  {
    "status": "success",
    "data": {
      "uid": "user_uid",
      "userEmail": "user@example.com",
      "userName": "John Doe"
    }
  }
  ```

### 2. Update User Data

- **Endpoint:** `PUT /user/updateUser`
- **Middleware:** `getResponseAuth`
- **Description:** Updates user data based on the provided parameters.
- **Request Body:**
  ```json
  {
    "uid": "user_uid",
    "email": "new_email@example.com",
    "name": "New Name",
    "data": {
      "gender": "Male",
      "height": 180,
      "weight": 75
    }
  }
  ```
- **Response:** `Data user_uid changed`

Feel free to contribute to this project by opening issues or submitting pull requests. Happy coding!
