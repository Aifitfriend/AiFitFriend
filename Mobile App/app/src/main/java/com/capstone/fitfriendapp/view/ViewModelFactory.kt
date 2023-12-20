package com.capstone.fitfriendapp.view

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.capstone.fitfriendapp.data.pref.UserRepository
import com.capstone.fitfriendapp.di.Injection
import com.capstone.fitfriendapp.view.home.MainViewModel
import com.capstone.fitfriendapp.view.login.LoginViewModel
import com.capstone.fitfriendapp.view.profile.ProfileViewModel
import com.capstone.fitfriendapp.view.register.RegisterViewModel

class ViewModelFactory (private val repository: UserRepository):
    ViewModelProvider.NewInstanceFactory() {

    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        return when {
            modelClass.isAssignableFrom(MainViewModel::class.java) -> {
                MainViewModel(repository) as T
            }

            modelClass.isAssignableFrom(LoginViewModel::class.java) -> {
                LoginViewModel(repository) as T
            }

            modelClass.isAssignableFrom(RegisterViewModel::class.java) -> {
                RegisterViewModel(repository) as T
            }

            modelClass.isAssignableFrom(ProfileViewModel::class.java) -> {
                ProfileViewModel(repository) as T
            }

            else ->
                throw IllegalArgumentException("Unknown ViewModel class : " +modelClass.name)
        }
    }

    companion object {
        @Volatile
        private var INSTANCE: ViewModelFactory? = null

        @JvmStatic
        fun getInstance(context: Context): ViewModelFactory =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: ViewModelFactory(Injection.provideRepository(context))
            }.also { INSTANCE = it }
    }
}