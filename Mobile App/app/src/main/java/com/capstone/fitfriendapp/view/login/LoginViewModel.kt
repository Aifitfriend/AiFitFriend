package com.capstone.fitfriendapp.view.login

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.capstone.fitfriendapp.data.pref.UserModel
import com.capstone.fitfriendapp.data.pref.UserRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

class LoginViewModel (private val repository: UserRepository): ViewModel(){

    fun login(email: String, password: String) = repository.login(email, password)

    fun saveSession(user: UserModel) {
        viewModelScope.launch {
            repository.saveSession(user)
        }
    }

    fun getSessionIsLogin(): Flow<UserModel> = repository.getSession()


}