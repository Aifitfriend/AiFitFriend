package com.capstone.fitfriendapp.view.register

import androidx.lifecycle.ViewModel
import com.capstone.fitfriendapp.data.pref.UserRepository

class RegisterViewModel (private val repository: UserRepository): ViewModel() {

    fun register (
        name: String,
        email: String,
        password: String
    ) = repository.register(name, email, password)

}