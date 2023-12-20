package com.capstone.fitfriendapp.view.home

import androidx.lifecycle.LiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.asLiveData
import androidx.lifecycle.viewModelScope
import com.capstone.fitfriendapp.data.pref.UserModel
import com.capstone.fitfriendapp.data.pref.UserRepository
import kotlinx.coroutines.launch

class MainViewModel (private val repository: UserRepository): ViewModel() {

    fun getSession(): LiveData<UserModel> {
        return repository.getSession().asLiveData()
    }

    fun logout() {
        viewModelScope.launch {
            repository.logout()
        }
    }

}