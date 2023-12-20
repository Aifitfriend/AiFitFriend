package com.capstone.fitfriendapp.di

import android.content.Context
import com.capstone.fitfriendapp.data.pref.UserPreference
import com.capstone.fitfriendapp.data.pref.UserRepository
import com.capstone.fitfriendapp.data.pref.dataStore
import com.example.fitfriend.data.api.ApiConfig
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking

object Injection {
    fun provideRepository(context: Context): UserRepository {
        val preference = UserPreference.getInstance(context.dataStore)
        val user = runBlocking { preference.getSession().first() }
        val apiService = ApiConfig.getApiService(user.token)
        return UserRepository.getInstance(preference, apiService)
    }
}