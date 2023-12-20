package com.capstone.fitfriendapp.data.pref

import retrofit2.HttpException
import androidx.lifecycle.LiveData
import androidx.lifecycle.liveData
import com.capstone.fitfriendapp.data.api.ApiService
import com.capstone.fitfriendapp.data.response.LoginResponse
import com.capstone.fitfriendapp.data.response.RegisterResponse
import com.google.gson.Gson
import kotlinx.coroutines.flow.Flow

class UserRepository private constructor(
    private val userPreference: UserPreference,
    private val apiService: ApiService
){
    fun getSession(): Flow<UserModel> {
        return userPreference.getSession()
    }

    suspend fun saveSession(user: UserModel) {
        userPreference.saveSession(user)
    }

    fun register (
        name: String,
        email: String,
        password: String
    ): LiveData<ResultState<RegisterResponse>> = liveData {
        emit(ResultState.Loading)
        try {
            val response = apiService.register(name, email, password)
            emit(ResultState.Success(response))
        } catch (e: HttpException) {
            emit(ResultState.Error("Registrasi Gagal"))
        }
    }

    fun login(email: String, password: String) = liveData {
        emit(ResultState.Loading)
        try {
            val response = apiService.login(email, password)
            emit(ResultState.Success(response))
        } catch (e: HttpException) {
            val errorBody = parseErrorBody<LoginResponse>(e)
            emit(ResultState.Error(errorBody.message))
        }
    }

    private inline fun <reified T> parseErrorBody(e: HttpException): T {
        val jsonInString = e.response()?.errorBody()?.string()
        return Gson().fromJson(jsonInString, T::class.java)
    }

    suspend fun logout() {
        userPreference.logout()
    }

    companion object {
        @Volatile
        private var instance: UserRepository? = null
        fun getInstance(
            userPreference: UserPreference,
            apiService: ApiService,
        ): UserRepository =
            instance ?: synchronized(this) {
                instance ?: UserRepository(userPreference, apiService)
            }.also { instance = it }
    }
}